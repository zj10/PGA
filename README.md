# PGA
A TensorFlow implementation of [perceptual generative autoencoder (PGA)](https://openreview.net/forum?id=rkxkr8UKuN), based on [this implementation](https://github.com/LynnHo/VAE-Tensorflow) of VAE.
## Usage
#### LPGA
```
# MNIST
train_lpga.py
# CIFAR-10
train_lpga.py --dataset cifar10 --model conv_32 --epoch 400 --lr 0.2 --bn True --z_dim 128 --zn_rec 3e-2
# CelebA
train_lpga.py --dataset celeba --model conv_64 --epoch 100 --lr 0.4 --bn True --z_dim 128 --zn_rec 3e-2 --zh_rec 1e-2 --nll 1e-2
```
#### VPGA
```
# MNIST
train_vpga.py
# CIFAR-10
train_vpga.py --dataset cifar10 --model conv_32 --epoch 400 --lr 0.2 --bn True --z_dim 128 --zn_rec 3e-2 --vrec 5e-3
# CelebA
train_vpga.py --dataset celeba --model conv_64 --epoch 60 --lr 0.4 --bn True --z_dim 128 --zn_rec 3e-2 --zh_rec 1e-2 --vkld 2e-3
```
