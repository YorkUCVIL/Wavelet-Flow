# Wavelet Flow: Fast Training of High Resolution Normalizing Flows
Code for *Wavelet Flow: Fast Training of High Resolution Normalizing Flows*.
Accpeted at NeurIPS 2020.

Project page with additional materials: <https://yorkucvil.github.io/Wavelet-Flow/>

Pre-trained weights, coming soon!

## Requirements
Developed on tensorflow 1.15 on Python 3.6.8. The code also expects tensorflow-probability 0.8.0 used for MCMC sampling. Addditional requirements are included in `requirements.txt`.

## Usage
Included under `src/scripts/` are a set of scripts for running Wavelet Flow. Each of these scripts run with a working directory of `src/`, independent of the working directory of the caller. If the `CUDA_VISIBLE_DEVICES` environment variable is already set, the GPU visibility arguments on the scripts, `-g`, will be ignored. The following examples for running the code assume the working directory is from `src/`. For every script, a dataset must be selected using `-d`.

The configuration for the models are located under `data/<DATASET NAME>/config.hjson` and settings such as hyper-parameters and paths. Included datasets are:
- imagenet_64
- imagenet_32
- celeba_1024
- ffhq_1024
- lsun_bedroom_64
- lsun_tower_64
- lsun_church_64

###### Training
```
python scripts/train.py -d lsun_bedroom_64 -g 0 -p 3
```
Here is an example command that can be run to train part of Wavelet Flow.
The dataset `lsun_bedroom_64` has been selected, while training on GPU 0, and training level 3 of the Wavelet Flow. All levels of a Wavelet Flow are trained independently; for `lsun_bedroom_64`, there are 7 levels which need to be trained with separate instances of the training script where `-p` is set to values 0-6.

###### Validation
```
python scripts/validate.py -d lsun_bedroom_64 -g 0
```
This is an example of how to compute the average BPD over the validation set for `lsun_bedroom_64` using GPU 0.

###### Sampling directly
```
python scripts/sample.py -d lsun_bedroom_64 -g 0 -n 5
```
This is an example of how to draw samples directly from the model without annealing. Samples in this example are output to `data/lsun_bedroom_64_haar/samples/` as a 5 by 5 grid of samples.

###### Annealed sampling with MCMC
```
python scripts/mcmc_sample.py -d lsun_bedroom_64 -g 0 -n 5 -t 0.97
```
This is an example of how to draw annealed samples from the model using MCMC. The arguments are similar to `sample.py` except there is a temperature parameter `-t` which specifies the annealing with the same convention as Glow. By default 10 MCMC iterations are used to perform step size adaptation, and 30 MCMC iterations are performed after fixing the step size before using taking a sample. Results are stored under `data/lsun_bedroom_64_haar/samples/`.

###### Super resolution
```
python scripts/super_res.py -d lsun_bedroom_64 -g 0 -n 5 -b 1
```
This is an example of how to draw super resolution samples. The parameters are similar to `sample.py` except `-b` should be specified to indicate the resolution of the low resolution image to be upsampled. Samples from the validation data are drawn and downsampled to the level specified by `-b` and upsampled using the Wavelet Flow model. Results are stored under `data/lsun_bedroom_64_haar/samples/`.

## Dataset setup
Datasets are expected to have a structure similar to the followinng example for `lsun_bedroom_64`:
```
lsun_bedroom_64
 |-- data
 |   |-- 64
 |       |-- 00000000.png
 |       |-- 00000001.png
 |       |-- 00000002.png
 |       .
 |       .
 |       .
 |       |-- 00099999.png
 |-- datalists
     |-- train_64.txt
     |-- val_64.txt
```
Images should be stored as PNGs, and the training and validation splits should be specified in their respective datalists. The datalists contain paths to each image on each line relative to the root of the dataset directory. By default, this code includes a dummy dataset which each model is configured to use.
