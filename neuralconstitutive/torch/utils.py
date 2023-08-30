from numbers import Real
from typing import Sequence

import torch
from torch import nn, Tensor


def beta(a: Tensor, b: Tensor) -> Tensor:
    # Adapted from https://github.com/rachtsingh/ibp_vae/blob/master/src/lgamma/beta.py#L22
    return (a.lgamma() + b.lgamma() - (a + b).lgamma()).exp()


def to_parameter(value: Real | Sequence) -> nn.Parameter:
    return nn.Parameter(torch.tensor(value, dtype=torch.float))
