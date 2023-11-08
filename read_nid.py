# %%
import mmap
from pathlib import Path
from typing import BinaryIO, ByteString, overload, Any, Self
from configparser import ConfigParser
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

NestedConfigDict = dict[str, dict[str, Any]]


@dataclass
class NIDChannel:
    index: int
    parent_index: int
    data: np.ndarray | None = None


@dataclass
class NIDGroup:
    index: int
    name: str
    id: int
    channels: list[NIDChannel]

    @property
    def count(self) -> int:
        """Returns the number of channels inside the group"""
        return len(self.channels)


@dataclass(frozen=True)
class NIDDataset(Sequence):
    version: int
    groups: list[NIDGroup]
    info: dict

    def __len__(self) -> int:
        """Returns the number of groups inside the dataset"""
        return len(self.groups)

    @overload
    def __getitem__(self, idx: int) -> NIDGroup:
        pass

    @overload
    def __getitem__(self, idx: slice) -> list[NIDGroup]:
        pass

    def __getitem__(self, idx):
        """Returns the NIDGroup element(s) corresponding to the idx"""
        return self.groups[idx]


def configparser_to_dict(config: ConfigParser) -> NestedConfigDict:
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)
    return config_dict


@dataclass
class NIDHeader:
    _header: dict[str, dict[str, Any]]

    @classmethod
    def from_bytes(cls, header: ByteString) -> Self:
        header_lines = header.decode().split("\r\n")

        # Read the ini format using configparser
        config = ConfigParser(allow_no_value=True, interpolation=None)
        config.optionxform = lambda option_name: option_name
        config.read_file(header_lines)

        config_dict = configparser_to_dict(config)
        return cls(config_dict)

    @property
    def dataset_config(self) -> dict[str, Any]:
        return self._header["DataSet"]

    @property
    def num_groups(self) -> int:
        """Returns the number of groups inside the nid file"""
        return int(self.dataset_config["GroupCount"])

    def get_group(self, group_idx: int) -> NIDGroup:
        if group_idx < 0 or group_idx >= self.num_groups:
            raise ValueError("Group index must be between 0 and num_groups - 1")

        # Get group name and id
        grp_name = self.dataset_config[f"Gr{group_idx}-Name"]
        grp_id = int(self.dataset_config[f"Gr{group_idx}-ID"])

        # Get the channels associated with the group
        max_channels = int(self.dataset_config[f"Gr{group_idx}-Count"])
        channels = []
        for channel_idx in range(max_channels):
            # Real number of channels is determined by the existance of the following key
            if f"Gr{group_idx}-Ch{channel_idx}" in self.dataset_config.keys():
                channels.append(NIDChannel(channel_idx, group_idx))

        return NIDGroup(group_idx, grp_name, grp_id, channels)

    def get_groups(self) -> list[NIDGroup]:
        return [self.get_group(i) for i in range(self.num_groups)]


def read_raw_header(fileio: BinaryIO, end_token: ByteString = b"#!") -> ByteString:
    header_end = fileio.find(end_token)
    body_start = header_end + len(end_token)
    header = fileio.read(header_end)
    fileio.seek(body_start)
    return header


# %%
filepath = Path("data/231106_onion_forcemap/Image02584.nid")
with open(filepath, "rb") as file:
    with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as memorymap:
        raw_header = read_raw_header(memorymap)
        header = NIDHeader.from_bytes(raw_header)

header.get_groups()
# %%


# %%
