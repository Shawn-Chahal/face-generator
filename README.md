# Face Generator

This repository trains a Wasserstein Generative Adversarial Network with Gradient Penalty on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset using [TensorFlow](https://github.com/tensorflow/tensorflow) in Python. The generator model is then used in the [Face Generator](https://www.shawnchahal.com/face-generator) web app.

## What does the app do?

[Face Generator](https://www.shawnchahal.com/face-generator) is a web app where the user is shown 64 faces which have been generated using a Generative Adversarial Network. The user can then select as many faces as they'd like to generate a new composite face.

## Training Time
celeba: 41 hours

flickr-faces: 

4x4: 9 h

8x8: 15 h (Total: 24 h)

16x16: 18 h (Total: 42 h)

32x32: 22 h (Total: 64 h)

64x64: 22+ h (Total: ?? h)

128x128: ?? h (Total: ?? h)
