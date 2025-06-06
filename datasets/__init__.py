# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.perm_mnist import PermutedMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.rot_mnist import RotatedMNIST
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.mnist_360 import MNIST360
from datasets.gcil_cifar100 import GCILCIFAR100
from datasets.seq_imagenet100 import SequentialImagenet100
from datasets.utils.continual_dataset import ContinualDataset
from datasets.seq_cifar100_vit import SequentialCIFAR100Vit
from datasets.seq_cifar10_vit import SequentialCIFAR10Vit
from argparse import Namespace

NAMES = {
    PermutedMNIST.NAME: PermutedMNIST,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    RotatedMNIST.NAME: RotatedMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MNIST360.NAME: MNIST360,
    GCILCIFAR100.NAME: GCILCIFAR100,
    SequentialImagenet100.NAME: SequentialImagenet100,
    SequentialCIFAR100Vit.NAME: SequentialCIFAR100Vit,
    SequentialCIFAR10Vit.NAME: SequentialCIFAR10Vit
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
