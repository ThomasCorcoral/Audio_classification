from src.load_data import AudiosetDataset
from src.generate_data import generate_lists
import sys
from datetime import datetime
import torch
from tqdm import tqdm
#sys.path.append('../misc/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')
from efficientnet_pytorch import EfficientNet
from src.EfficientNet import EffNetAttention
from src.diffres import DiffRes
import torch.nn as nn

from src.generate_data import generate_lists

import os
import sys
import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from torch.utils.tensorboard import SummaryWriter # Used to see training evolution
import gc # Garbage collector

PATH_DATASET = "./datasets/speechcommands"
PATH_DATA = "./misc/diffres_data_speechcommands"
PATH_OUTPUT = "./working/"

if __name__ == "__main__":

    generate_lists(PATH_DATASET, PATH_OUTPUT, PATH_DATA)

    data_train = PATH_OUTPUT + 'datafiles/speechcommand_train_data.json'
    data_val = PATH_OUTPUT + 'datafiles/speechcommand_valid_data.json'
    data_eval = PATH_OUTPUT + 'datafiles/speechcommand_eval_data.json'

    exp_dir = "./working/save_model/"

    num_mel_bins = 128
    target_length = 98

    audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length}
    
    val_audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length}

    label_csv = PATH_DATA + "/speechcommands_class_labels_indices.csv"
    hop_ms = 10
    batch_size = 128
    num_workers = 2 # Max recommended is 2

    dataset = AudiosetDataset(data_train,
                            label_csv=label_csv,
                            audio_conf=audio_conf,
                            hop_ms=hop_ms)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=num_workers,
                                            #sampler=None,
                                            pin_memory=False, 
                                            drop_last=True,
                                            generator=g)

    dataset_val = AudiosetDataset(data_val, 
                                label_csv=label_csv, 
                                audio_conf=val_audio_conf, 
                                hop_ms=hop_ms)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                            batch_size=batch_size // 2, 
                                            shuffle=False, 
                                            num_workers=num_workers,
                                            pin_memory=True, 
                                            drop_last=True, 
                                            generator=g)

    n_class = 35 # 35 classes in speechcommands
    eff_b = 2 # To select efficientnet network
    impretrain = True # Use pretrained model or not
    att_head = 6 # Number of attentions heads
    target_length = 98
    preserve_ratio = 0.15
    alpha = 1.0
    learn_pos_emb = False
    use_leaf = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    print(device)

    audio_model = EffNetAttention(label_dim=n_class, 
                                b=eff_b, 
                                pretrain=impretrain, 
                                head_num=att_head, 
                                input_seq_length=target_length,
                                sampler=eval("DiffRes"), 
                                preserve_ratio=preserve_ratio, 
                                alpha=alpha, 
                                learn_pos_emb=learn_pos_emb,
                                n_mel_bins=num_mel_bins).to(device)

    epoch = 0
    n_epochs = 20
    global_step = 0

    if(os.path.exists(os.path.join(exp_dir, "audio_model.pth"))):
        model_checkpoint = torch.load(os.path.join(exp_dir, "audio_model.pth"), map_location="cpu")
        audio_model.load_state_dict(model_checkpoint["state_dict"])
        epoch = model_checkpoint["epoch"]
        global_step = model_checkpoint["global_step"]

    audio_model.to(device)

    lr=2.5e-3

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))

    if(os.path.exists(os.path.join(exp_dir, "optim_state.pth"))):
        opt_checkpoint = torch.load(os.path.join(exp_dir, "optim_state.pth"), map_location="cpu")
        optimizer.load_state_dict(opt_checkpoint["state_dict"])
        epoch = model_checkpoint["epoch"]
        global_step = model_checkpoint["global_step"]

    lrscheduler_start=3
    lrscheduler_decay=0.7

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(lrscheduler_start, 1000, 1)), gamma=lrscheduler_decay, last_epoch=epoch-1)

    #loss_fn = nn.BCELoss(reduction="none")
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter()
    audio_model.train()

    while epoch < n_epochs + 1:
        print("Epoch:", epoch)
        audio_model.train()
        for i, (audio_input, labels, fnames) in enumerate(tqdm(train_loader)):
            #B = audio_input.size(0)
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
            if global_step % 100 == 0:
                loss_test = []
                audio_model.eval() # Now evaluate the model
                true_pred, false_pred = 0, 0
                with torch.no_grad():
                    # Get the data from the val_loader
                    for i, (audio_input, labels, fname) in enumerate(tqdm(val_loader)):  
                        #batchsize = audio_input.size(0)
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
                        #del audio_input, audio_output, labels_np, current_true

                accuracy_test = true_pred / (true_pred + false_pred)
                
                # Writing to tensorboard
                writer.add_scalar('Loss/train', loss.item(), global_step/100)
                writer.add_scalar('Loss/test', np.mean(loss_test), global_step/100)
                writer.add_scalar('Accuracy/test', accuracy_test, global_step/100)

                del loss_test, accuracy_test
                gc.collect()

                audio_model.train()

            global_step += 1
            # gc.collect()

        scheduler.step()

        epoch += 1

        torch.save(audio_model.state_dict(), "./working/save_model/model.pt")
        torch.save({"state_dict": audio_model.state_dict(), "epoch": epoch, "global_step": global_step}, "%saudio_model.pth" % (exp_dir))
        torch.save({"state_dict": optimizer.state_dict(), "epoch": epoch, "global_step": global_step}, "%soptim_state.pth" % (exp_dir))
