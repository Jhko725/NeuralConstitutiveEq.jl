from typing import TypedDict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class IndentationDataBatch(TypedDict):
    time: Tensor
    indent: Tensor
    velocity: Tensor
    force: Tensor


class IndentationDataset(Dataset):
    def __init__(
        self, time: Tensor, indentation: Tensor, velocity: Tensor, force: Tensor
    ):
        # May want to check the shapes of the tensors
        # Expect all of them to be (1, n_time)
        # Later, may generalize to (n_samples, n_time)
        super().__init__()
        self.time = torch.tensor(time).view(1, -1)
        self.indent = torch.tensor(indentation).view(1, -1)
        self.velocity = torch.tensor(velocity).view(1, -1)
        self.force = torch.tensor(force).view(1, -1)

    def __len__(self):
        return self.time.shape[0]

    def __getitem__(self, ind) -> IndentationDataBatch:
        return {
            "time": self.time[ind],
            "indent": self.indent[ind],
            "velocity": self.velocity[ind],
            "force": self.force[ind],
        }


def split_app_ret(
    dataset: IndentationDataset,
) -> tuple[IndentationDataset, IndentationDataset]:
    time, indent, velocity, force = (
        dataset.time,
        dataset.indent,
        dataset.velocity,
        dataset.force,
    )
    ind_tmax = torch.argmax(indent, dim=-1) + 1
    dataset_app = IndentationDataset(
        time[..., :ind_tmax],
        indent[..., :ind_tmax],
        velocity[..., :ind_tmax],
        force[..., :ind_tmax],
    )
    dataset_ret = IndentationDataset(
        time[..., ind_tmax:],
        indent[..., ind_tmax:],
        velocity[..., ind_tmax:],
        force[..., ind_tmax:],
    )
    return dataset_app, dataset_ret
