import torch, os, gc
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter # Used to see training evolution

from src.load_data import AudiosetDataset
from src.generate_data import generate_lists
from src.EfficientNet import EffNetAttention
from src.diffres import DiffRes, AvgPool, ConvAvgPool, AvgMaxPool, ChangeHopSize

PATH_DATASET = "./datasets/speechcommands"
PATH_DATA = "./misc/diffres_data_speechcommands"
PATH_OUTPUT = "./working/"

def prepare_start():
    generate_lists(PATH_DATASET, PATH_OUTPUT, PATH_DATA)

    data_train = PATH_OUTPUT + 'datafiles/speechcommand_train_data.json'
    data_val = PATH_OUTPUT + 'datafiles/speechcommand_valid_data.json'
    #data_eval = PATH_OUTPUT + 'datafiles/speechcommand_eval_data.json'

    return data_train, data_val

def prepare_parameters(algo):

    params = {}

    params["exp_dir"] = "./working/save_model/"

    params["num_mel_bins"] = 128
    params["target_length"] = 98

    params["audio_conf"] = {'num_mel_bins': params["num_mel_bins"], 'target_length': params["target_length"]}

    params["label_csv"] = PATH_DATA + "/speechcommands_class_labels_indices.csv"
    params["hop_ms"] = 10 # how much each window moves forward relative to the previous one
    params["batch_size"] = 32 # Size of each batch
    params["num_workers"] = 2 # Max recommended is 2

    params["n_class"] = 35 # 35 classes in speechcommands
    params["eff_b"] = 2 # To select efficientnet network
    params["pretrained"] = True # Use pretrained model or not
    params["att_head"] = 4 # Number of attentions heads
    params["preserve_ratio"] = 0.4 # will preserve x% of original audio
    params["alpha"] = 1.0
    params["learn_pos_emb"] = False

    # params["algo"] = "DiffRes" # Drop-in algorithm : DiffRes, AvgPool, ConvAvgPool, AvgMaxPool, ChangeHopSize
    params["algo"] = algo

    params["epoch"] = 0 # Current epoch
    params["n_epochs"] = 20 # Number of epoch to stop training
    params["global_step"] = 0 # 1 step = 1 batch

    params["lr"] = 2.5e-3 # learning rate
    params["lrscheduler_start"] = 3 # When to start decreasing the lr
    params["lrscheduler_decay"] = 0.7 # coefficient of reduction

    return params

def prepare_data(data, label_csv, audio_conf, hop_ms, batch_size, num_workers, train=True, mfcc=False):

    if not train:
        bs = batch_size // 2
    else:
        bs = batch_size

    dataset = AudiosetDataset(data,
                            label_csv=label_csv,
                            audio_conf=audio_conf,
                            hop_ms=hop_ms,
                            mfcc=mfcc)

    loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=bs, 
                                            shuffle=True, 
                                            num_workers=num_workers,
                                            #sampler=None,
                                            pin_memory=False, 
                                            drop_last=True)
    return loader

def get_model(n_class, eff_b, pretrained, att_head, target_length, algo, preserve_ratio, alpha, learn_pos_emb, num_mel_bins, device, ratio_visualize):
    if algo is None:
        a = None
    else:
        a = eval(algo)
    model = EffNetAttention(label_dim=n_class, 
                            b=eff_b, 
                            pretrain=pretrained, 
                            head_num=att_head, 
                            input_seq_length=target_length,
                            sampler=a, 
                            preserve_ratio=preserve_ratio, 
                            alpha=alpha, 
                            learn_pos_emb=learn_pos_emb,
                            n_mel_bins=num_mel_bins,
                            ratio_visualize=ratio_visualize,
                            device=device).to(device)
    return model

def run(algo, mfcc):

    data_train, data_eval = prepare_start()

    params = prepare_parameters(algo)

    ratio_save = 128/params["batch_size"]

    train_loader = prepare_data(data_train, params["label_csv"], params["audio_conf"], params["hop_ms"], params["batch_size"], params["num_workers"], mfcc=mfcc)
    val_loader = prepare_data(data_eval, params["label_csv"], params["audio_conf"], params["hop_ms"], params["batch_size"], params["num_workers"], train=False, mfcc=mfcc)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    print(device)

    audio_model = get_model(params["n_class"], params["eff_b"], params["pretrained"], params["att_head"], params["target_length"], params["algo"], params["preserve_ratio"], params["alpha"], params["learn_pos_emb"], params["num_mel_bins"], device, ratio_save)

    epoch = params["epoch"]
    global_step = params["global_step"]

    if(os.path.exists(os.path.join(params["exp_dir"], "audio_model.pth"))):
        model_checkpoint = torch.load(os.path.join(params["exp_dir"], "audio_model.pth"), map_location="cpu")
        audio_model.load_state_dict(model_checkpoint["state_dict"])
        epoch = model_checkpoint["epoch"]
        global_step = model_checkpoint["global_step"]

    audio_model.to(device)

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, params["lr"], weight_decay=5e-7, betas=(0.95, 0.999))

    if(os.path.exists(os.path.join(params["exp_dir"], "optim_state.pth"))):
        opt_checkpoint = torch.load(os.path.join(params["exp_dir"], "optim_state.pth"), map_location="cpu")
        optimizer.load_state_dict(opt_checkpoint["state_dict"])
        epoch = model_checkpoint["epoch"]
        global_step = model_checkpoint["global_step"]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(params["lrscheduler_start"], 1000, 1)), gamma=params["lrscheduler_decay"], last_epoch=epoch-1)

    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter()
    audio_model.train()

    while epoch < params["n_epochs"] + 1:
        print("Epoch: ", epoch)
        audio_model.train()
        for _, (audio_input, labels) in enumerate(tqdm(train_loader)):
            
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            audio_output, score_pred = audio_model(audio_input)

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                epsilon = 1e-7
                audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
                loss = loss_fn(audio_output, labels)
                loss = torch.mean(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del audio_input, audio_output, score_pred

            # Eval the model every 100 batches
            if global_step % (ratio_save*100) == 0:
                loss_test = []
                audio_model.eval() # Now evaluate the model
                true_pred, false_pred = 0, 0
                with torch.no_grad():
                    # Get the data from the val_loader
                    for _, (audio_input, labels) in enumerate(tqdm(val_loader)):  
                        
                        audio_input = audio_input.to(device)  
                        audio_output,_ = audio_model(audio_input) 
                        predictions_np = audio_output.to('cpu').detach().numpy()
                        labels_np = labels.to('cpu').detach().numpy()
                        current_true = np.sum([1 for i in range(len(predictions_np)) if np.argmax(predictions_np[i]) == np.argmax(labels_np[i])])
                        true_pred += current_true
                        false_pred += len(predictions_np) - current_true
                        epsilon = 1e-7
                        audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
                        ll = loss_fn(audio_output.to('cpu'), labels.to('cpu'))
                        ll = torch.mean(ll)
                        loss_test.append(ll.item())

                accuracy_test = true_pred / (true_pred + false_pred)
                
                # Writing to tensorboard
                writer.add_scalar('Loss/train', loss.item(), global_step/(ratio_save*100))
                writer.add_scalar('Loss/test', np.mean(loss_test), global_step/(ratio_save*100))
                writer.add_scalar('Accuracy/test', accuracy_test, global_step/(ratio_save*100))

                del loss_test, accuracy_test
                gc.collect()

                audio_model.train()

            global_step += 1

        scheduler.step()
        epoch += 1
        torch.save(audio_model.state_dict(), "./working/save_model/model.pt")
        torch.save({"state_dict": audio_model.state_dict(), "epoch": epoch, "global_step": global_step}, "%saudio_model.pth" % (params["exp_dir"]))
        torch.save({"state_dict": optimizer.state_dict(), "epoch": epoch, "global_step": global_step}, "%soptim_state.pth" % (params["exp_dir"]))
    return 0

if __name__ == "__main__":
    #run(None, False)
    run("DiffRes", False)
