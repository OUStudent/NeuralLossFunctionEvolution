# NeuralLossFunctionEvolution

# Reproducing Results

To keep things simple, we will reproduce results using the ResNet9V2 80x80 resolution surrogate function.

- First, run ``loss_function_evolution.py`` to run the evolutionary search (This one is not timmed, so it runs for 4,000 iterations)
- Second, manually create a list of the best losses found over the course of evolutoin, and pickle it
- Third, run `loss_projection.py` to reduce the potential losses down to the best 12
- Fourth, manually create a list of the best losses that you chose from the best 12, found from the full scale RandAug training strategy, and pickle it.
- Fifth, run `final_loss_testing.py` to test the final losses on either Cifar10 or Cifar100
