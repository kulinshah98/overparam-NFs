# Learning and Generalization in Overparameterized Normalizing Flows

This repository contains a pytorch code to replicate experiments in the paper: [Learning and Generalization in Overparameterized Normalizing Flows](https://arxiv.org/abs/2106.10535).

## Requirements

The code for synthetic datasets is tested on
- python 3.6.9
- pytorch 1.10.1
- matplotlib 3.3.4
- numpy 1.19.5

## Overview

- Experiments of the paper is divided in two parts. Code files for Constrained Normalizing Flows (CNFs) are given in `./CNF/` folder. Code files for Unconstrained Normalizing Flows (UNFs) are given in `./UNF/` folder.
- All synthetic datasets used in the paper are given in `./datasets/` folder.
- Code to reproduce the results for the Miniboone dataset is given in `./BNAF/` and `./UMNN/`.

## Acknowledgement

For experiments on Miniboone datasets, we use the code from [BNAF](https://github.com/nicola-decao/BNAF) and [UMNN](https://github.com/AWehenkel/UMNN).

## Citation

If you find this project useful, please consider citing the following publication:

```
@article{shah2021learning,
  title={Learning and Generalization in Overparameterized Normalizing Flows},
  author={Shah, Kulin and Deshpande, Amit and Goyal, Navin},
  journal={arXiv preprint arXiv:2106.10535},
  year={2021}
}
```
