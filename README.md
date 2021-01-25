## General Info
This repository includes code for training a Convolutional Neural Network (CNN) to identify 4 classes of Interferences namely: CI, CWI, MCWI, SOI.

We use two CNN's to classify interferences and their accuracy results are as follows:
1. Use Pretrained ResNet18 model - 97.8% accuracy
2. Custom CNN Model - 85% Accuracy

This implementation uses pytorch library and pytorch lightning trainer

## Requirements
Python3, Pytorch, Pytorch Lightning

## Setup
To run this project install the requirements and make sure you have at least 14GB GPU:
```
bash train.sh
```

## Todo
1. Use command line arguments to change training parameters e.g batch size and epochs
2. Remove print logs
3. Add comments