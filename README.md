The code for the paper 
[**Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes**](https://arxiv.org/abs/2209.03695), NeurIPS'22

Create env:
```(bash)
conda env create -f SI_regimes_env.yml
```

Example usage - how to obtain one of the lines at figure 1 in the paper:
1. Run script run_train_and_test.py to train and compute metrics (in the presented form it traines a scale-invariant ConvNet on CIFAR-10 dataset trained using SGD on the sphere with ELR 1e-3).
2. Use notebook Plots.ipynb to look at the results.

Main parameters to change in run_train_and_test.py to replicate other results:
- dataset: CIFAR10 or CIFAR100
- to train fully scale-invariant networks use models ConvNetSI/ResNet18SI, fix_noninvlr = 0.0 (learning rate for not scale invariant parameters) and initscale = 10. (norm of the last layer weight matrix)
- to train all network parameters use models ConvNetSIAf/ResNet18SIAf, fix_noninvlr = -1 and initscale = -1
- to train networks on the sphere use fix_elr = 'fix_elr' and some positive elr value
- to train network in the whole parameter space use fix_elr = 'fix_lr' and some positive lr_init value (+ we use weight decay wd in this setup)
- to turn on the momentum use a non-zero value for it in params
- to turn on data augmentation delete noaug option from add_params
