# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.z

import random
import torch
import numpy as np


def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return 'data/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
