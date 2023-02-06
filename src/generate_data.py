import os
import numpy as np
from tqdm import tqdm
import json

def generate_lists(PATH_DATASET, PATH_OUTPUT, PATH_DATA):
    # generate training list = all samples - validation_list - testing_list
    if os.path.exists(os.path.join(PATH_DATASET, 'train_list.txt'))==False:
        with open(os.path.join(PATH_DATASET, 'validation_list.txt'), 'r') as f:
            val_list = f.readlines()

        with open(os.path.join(PATH_DATASET, 'testing_list.txt'), 'r') as f:
            test_list = f.readlines()

        val_test_list = list(set(test_list+val_list))

        def get_immediate_subdirectories(a_dir):
            return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
        def get_immediate_files(a_dir):
            return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

        base_path = PATH_DATASET
        all_cmds = get_immediate_subdirectories(base_path)
        all_list = []
        for cmd in tqdm(all_cmds):
            if cmd != '_background_noise_':
                cmd_samples = get_immediate_files(base_path+'/'+cmd)
                for sample in cmd_samples:
                    all_list.append(cmd + '/' + sample+'\n')

        training_list = [x for x in all_list if x not in val_test_list]

        with open(os.path.join(PATH_OUTPUT, 'train_list.txt'), 'w') as f:
            f.writelines(training_list)

    label_set = np.loadtxt(PATH_DATA + "/speechcommands_class_labels_indices.csv", delimiter=',', dtype='str')
    label_map = {}
    for i in range(1, len(label_set)):
        label_map[eval(label_set[i][2])] = label_set[i][0]

    os.makedirs(PATH_OUTPUT + 'datafiles', exist_ok=True)
    base_path = PATH_DATASET + "/"
    for split in ['testing', 'validation', 'train']:
        if split == 'train':
            current_path = PATH_DATASET +'/'+ split + '_list.txt'
        else:
            current_path = base_path + split + '_list.txt'
        wav_list = []
        with open(current_path, 'r') as f:
            filelist = f.readlines()
        for file in tqdm(filelist):
            cur_label = label_map[file.split('/')[0]]
            cur_path = os.path.join(PATH_DATASET, file.strip()) 
            cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
            wav_list.append(cur_dict)
        np.random.shuffle(wav_list)
        if split == 'train':
            with open(PATH_OUTPUT + 'datafiles/speechcommand_train_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'testing':
            with open(PATH_OUTPUT + 'datafiles/speechcommand_eval_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'validation':
            with open(PATH_OUTPUT + 'datafiles/speechcommand_valid_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))