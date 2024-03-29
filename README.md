# Audio_classification
Learning the spectrogram temporal resolution for audio classification

Based on paper : Anonymous (2023). LEARNING THE SPECTROGRAM TEMPORAL RESOLUTION FOR AUDIO CLASSIFICATION. In Submitted to The Eleventh International Conference on Learning Representations (https://openreview.net/forum?id=HOF3CTk2WH6)

## 1. Installation

### 1.0 Download the dataset 

First, we will use the speechcommands dataset. This dataset contains +105,000 sounds for 35 classes. The audios are short and clear. Here is the link to download the dataset : https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Extract the data in the ./datasets/ folder. Now you have a folder nammed 'speechcommands' inside the folder 'datasets'

You will also need to create some folders : './runs/' './working/'

### 1.1 Prepare the environment

```shell
# Create a new conda environment diffres
conda env create --name diffres --file=env.yml
```

Activate the environment

```shell
conda activate diffres
```

### 1.2 Start training

If you don't want to train, you can directly go to step 1.3 and start testing the model

To train the model, simply run the main.py file as 

```shell
python3 main.py
```

If you want to specify the preserve ratio and the batch size directly in the shell, you can use the following arguments

```shell
python3 main.py --ratio 0.5 --batch-size 128
```


If you want to modify other parameters, juste edit the main.py file.
