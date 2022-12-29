# NeuralLossFunctionEvolution

# Reproducing Results

To keep things simple, we will reproduce results using the ResNet9V2 80x80 resolution surrogate function.

- First, run ``loss_function_evolution.py`` to run the evolutionary search (This one is not timmed, so it runs for 4,000 iterations)
- Second, manually create a list of the best losses found over the course of evolutoin, and pickle it
- Third, run `loss_projection.py` to reduce the potential losses down to the best 12
- Fourth, manually create a list of the best losses that you chose from the best 12, found from the full scale RandAug training strategy, and pickle it.
- Fifth, run `final_loss_testing.py` to test the final losses on either Cifar10 or Cifar100

# Repository Overview

## ``effnet_rand_aug.py``

This file contains the code derived from https://github.com/leondgarse/keras_efficientnet_v2 for creating EffNetV2 along with progressive RandAug reguralization

## ``fitness_functions.py``

This file contains the two surrogate functions, ResNetV2 80x80 and EffNetV2 96x96, used as the fitness functions during the course of evolution. In addition, the simple ConvNet and ResNetV2 without RandAug regularization are included as well.

## ``loss_function_evolution.py``

This file performs the neural loss function evolution through the mutation only regularized evolutionary algorithm. 

## ``loss_projection``

This file scales the best loss functions found after the evolutionary search, eliminating them using the strategy proposed in the paper.

## ``final_loss_testing.py`` 

This file takes the final selected loss functions and trains each one 6 times on EffNetV2 with different parameters: with/without pre-trained ImageNet weights and/or progressive RandAug regularization.
