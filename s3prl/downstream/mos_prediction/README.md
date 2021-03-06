# MOS Prediction

Official Implementation of "Utilizing Self-supervised Representations for MOS Prediction", will be presented at INTERSPEECH 2021 [[arXiv](https://arxiv.org/abs/2104.03017)]

This code enables you to fine-tune a automatic Mean Opinion Score (MOS) Predictor with specific self-supervised upstream model.

## Pretrained MOS Predictor
If you only want to directly use the pretrained MOS predictor instead of fine-tuning your own one, please refer to the MOS Prediction upstream (See [**MOS Predictor**](../../upstream/mos_prediction)).

## Data Preparation
Download Voice Conversion Challenge 2018 (VCC 2018) dataset and Voice Conversion Challenge 2016 (VCC 2016) dataset:
```bash
cd /path/to/data

# Download VCC 2018
mkdir VCC_2018
cd VCC_2018
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_evaluation_results.zip
tar -zxvf vcc2018_submitted_systems_converted_speech.tar.gz
unzip vcc2018_evaluation_results.zip
cd ..

# Download VCC 2016
mkdir VCC_2016
cd VCC_2016 
wget https://datashare.ed.ac.uk/bitstream/handle/10283/2211/participant_submissions.zip
unzip participant_submissions.zip
```
Then put the corresponding .csv file in [**data**](data) under each dataset folder

After that, you should have the following file structure:
```bash
   /path/to/data
   ├── VCC_2018
   │   ├── Converted_speech_of_submitted_systems
   │   │   └── [*.wav]
   │   ├── VCC2018_Results
   │   └── [train, valid, test]_judge.csv
   │
   └── VCC_2016
       ├── unified_speech
       │   └── [*.wav]
       └── system_mos.csv
```




## Train a New Model
Use the following code to train a MOS Predictor with specific upstream model:
```bash
EXP_NAME=hello_world
UPSTREAM=wav2vec2
DOWNSTREAM=mos_prediction

python3 run_downstream.py -f -m train -n $EXP_NAME -u $UPSTREAM -d $DOWNSTREAM
```

## Customize Your Own Model
You can also customize your own model with specfic dataset or downstream structure by modifying the code (See [**Add new downstream tasks**](../../downstream/README.md#add-new-downstream-tasks)). 

## Citation

If you find this MOS predictor useful, please consider citing following paper:
```
@article{tseng2021utilizing,
  title={Utilizing Self-supervised Representations for MOS Prediction},
  author={Tseng, Wei-Cheng and Huang, Chien-yu and Kao, Wei-Tsung and Lin, Yist Y and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2104.03017},
  year={2021}
}
```
