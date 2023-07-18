## Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes

This repo contains the official PyTorch implementation of the NeurIPS'22 paper  

**Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes**  
[Maxim Kodryan](https://scholar.google.com/citations?user=BGVWciMAAAAJ&hl=en)\*, 
[Ekaterina Lobacheva](https://tipt0p.github.io/)\*, 
[Maksim Nakhodnov](https://www.linkedin.com/in/nakhodnov17/?originalSubdomain=ru)\*,
[Dmitry Vetrov](https://scholar.google.com/citations?user=7HU0UoUAAAAJ&hl=en)

[arXiv](https://arxiv.org/abs/2209.03695) / 
[openreview](https://openreview.net/forum?id=edffTbw0Sws) /
[short poster video](https://nips.cc/virtual/2022/poster/53275) / 
[long talk (in Russian)](https://www.youtube.com/watch?v=ZCIa6HuawQY) / 
[bibtex](https://tipt0p.github.io/papers/three_regimes_neurips22.txt)

## Abstract
<div align="justify">
<img align="right" width=40% src="https://tipt0p.github.io/images/three_regimes_neurips22.png" />
A fundamental property of deep learning normalization techniques, such as batch
normalization, is making the pre-normalization parameters scale invariant. The
intrinsic domain of such parameters is the unit sphere, and therefore their gradient
optimization dynamics can be represented via spherical optimization with varying
effective learning rate (ELR), which was studied previously. However, the varying
ELR may obscure certain characteristics of the intrinsic loss landscape structure. In
this work, we investigate the properties of training scale-invariant neural networks
directly on the sphere using a fixed ELR. We discover three regimes of such training
depending on the ELR value: convergence, chaotic equilibrium, and divergence.
We study these regimes in detail both on a theoretical examination of a toy example
and on a thorough empirical analysis of real scale-invariant deep learning models.
Each regime has unique features and reflects specific properties of the intrinsic
loss landscape, some of which have strong parallels with previous research on
both regular and scale-invariant neural networks training. Finally, we demonstrate
how the discovered regimes are reflected in conventional training of normalized
networks and how they can be leveraged to achieve better optima. 
</div>


## Code

**Environment**
```(bash)
conda env create -f SI_regimes_env.yml
```

**Example usage**  
To obtain one of the lines in Figure 1 in the paper:
1. Run script run_train_and_test.py to train and compute metrics (in the presented form, it trains a scale-invariant ConvNet on CIFAR-10 using SGD on the sphere with ELR 1e-3).
2. Use notebook Plots.ipynb to look at the results.

**Main parameters**  
To replicate other results from the paper, vary the parameters in run_train_and_test.py:
- dataset: CIFAR10 or CIFAR100
- to train fully scale-invariant networks use models ConvNetSI/ResNet18SI, fix_noninvlr = 0.0 (learning rate for not scale invariant parameters), and initscale = 10. (norm of the last layer weight matrix)
- to train all network parameters use models ConvNetSIAf/ResNet18SIAf, fix_noninvlr = -1 and initscale = -1
- to train networks on the sphere use fix_elr = 'fix_elr' and some positive elr value
- to train network in the whole parameter space use fix_elr = 'fix_lr' and some positive lr_init value (+ we use weight decay wd in this setup)
- to turn on the momentum use a non-zero value for it in params
- to turn on data augmentation delete the noaug option from add_params

## Attribution

Parts of this code are based on the following repositories:
- [On the Periodic Behavior of Neural Network Training with Batch Normalization and Weight Decay](https://github.com/tipt0p/periodic_behavior_bn_wd). Ekaterina Lobacheva, Maxim Kodryan, Nadezhda Chirkova, Andrey Malinin, and Dmitry Vetrov.
- [Rethinking Parameter Counting: Effective Dimensionality Revisted](https://github.com/g-benton/hessian-eff-dim). Wesley Maddox, Gregory Benton, and Andrew Gordon Wilson.

## Citation

If you found this code useful, please cite our paper
```
@inproceedings{kodryan2022regimes,
    title={Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes},
    author={Maxim Kodryan and Ekaterina Lobacheva and Maksim Nakhodnov and Dmitry Vetrov},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022},
    url={https://openreview.net/forum?id=edffTbw0Sws}
}
```
