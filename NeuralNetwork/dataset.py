from typing import TypedDict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class IndentationDataBatch(TypedDict):
    time: Tensor
    indent: Tensor
    force: Tensor


class IndentationDataset(Dataset):
    def __init__(self, time: Tensor, indentation: Tensor, force: Tensor):
        # May want to check the shapes of the tensors
        # Expect all of them to be (n_samples, n_time)
        super().__init__()
        self.time = torch.tensor(time)
        self.indent = torch.tensor(indentation)
        self.force = torch.tensor(force)

    def __len__(self):
        return self.time.shape[0]

    def __getitem__(self, ind) -> IndentationDataBatch:
        return {"time": self.time[ind], "force": self.force[ind]}


def split_app_ret(
    dataset: IndentationDataset,
) -> tuple[IndentationDataset, IndentationDataset]:
    time, indent, force = dataset.time, dataset.indent, dataset.force
    ind_tmax = torch.argmax(indent, dim=-1) + 1
    dataset_app = IndentationDataset(
        time[..., :ind_tmax], indent[..., :ind_tmax], force[..., :ind_tmax]
    )
    dataset_ret = IndentationDataset(
        time[..., ind_tmax:], indent[..., ind_tmax:], force[..., ind_tmax:]
    )
    return dataset_app, dataset_ret
