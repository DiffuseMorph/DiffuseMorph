# DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model

This repository is the official implementation of "DiffuseMorph: Unsupervised Deformable Image Registration Using Diffusion Model".
<img src="./img/representative.png"  height="400">

## Requirements
  * OS : Ubuntu
  * Python >= 3.6
  * PyTorch >= 1.4.0

## Data
In our experiments, we used the following datasets:
* 2D facial expression images: [RaFD dataset](https://rafd.socsci.ru.nl/RaFD2/RaFD?p=main)
* 3D cardiac MR images: [ACDC dataset](https://acdc.creatis.insa-lyon.fr/description/databases.html)
* 3D brain MR images: [OASIS-3 dataset](https://www.oasis-brains.org/)

## Training

To train our model for 2D image registration, run this command:

```train
python3 main_2D.py -p train -c config/diffuseMorph_train_2D.json
```
To train our model for 3D image registration, run this command:

```train
python3 main_3D.py -p train -c config/diffuseMorph_train_3D.json
```

## Test

To test the trained our model for 2D image registration, run:

```eval
python3 main_2D.py -p test -c config/diffuseMorph_test_2D.json
```

To test the trained our model for 3D image registration, run:

```eval
python3 main_3D.py -p test -c config/diffuseMorph_test_3D.json
```

## Pre-trained Models

You can download our pretrained model for 2D image registration [here](https://drive.google.com/drive/folders/1-caDkoMI_u7sNJeIGWrlxSQWdx2hKCvU?usp=sharing).
Then, you can test the model by saving the pretrained weights in the directory ./checkpoints.
To brifely test our method given the pretrained model, we provided the toy example in the directory './toy_sample'.

