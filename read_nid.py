# %%
import mmap
from pathlib import Path
from typing import BinaryIO, ByteString, overload, Any, Self
from configparser import ConfigParser
from dataclasses import dataclass
from collections.abc import Sequence
import re

import numpy as np

NestedConfigDict = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class NIDChannelHeader:
    _channel_header: dict[str, Any]

    @property
    def name(self) -> str:
        return self._channel_header["Frame"]

    @property
    def dtype(self) -> np.dtype:
        saveorder = self._channel_header["SaveOrder"]
        savesign = self._channel_header["SaveSign"]
        savebits = int(self._channel_header["SaveBits"])

        endian = "<" if saveorder == "Intel" else ">"
        data_format = "i" if savesign == "Signed" else "u"
        num_bytes = savebits // 8
        return np.dtype(f"{endian}{data_format}{num_bytes}")

    @property
    def shape(self) -> tuple[int, int]:
        num_x = int(self._channel_header["Points"])
        num_y = int(self._channel_header["Lines"])
        return (num_x, num_y)

    def sizeof(self) -> int:
        """Returns the size of the data associated with the channel in bytes"""
        bytes_per_point = self.dtype.itemsize
        n_pixels = self.shape[0] * self.shape[1]
        return n_pixels * bytes_per_point
        # if not self.is_jagged():
        #     n_pixels = self.shape[0] * self.shape[1]
        #     return n_pixels * bytes_per_point
        # else:
        #     n_jags = self.shape[1]
        #     n_data = sum(
        #         int(self._channel_header[f"LineDim{i}Points"]) for i in range(n_jags)
        #     )
        #     return n_data * bytes_per_point

    def is_jagged(self) -> bool:
        """Returns whether if the channel represents normal array/tensor like data
        or if it represents a jagged/ragged array/tensor, which is usually created though
        measurements that terminate/trigger though a setpoint value."""

        header_keys = self._channel_header.keys()
        return any(
            bool(re.match(r"LineDim\d+(?:Range|Min|Points)", k)) for k in header_keys
        )


@dataclass
class NIDChannel:
    index: int
    parent_index: int
    header: NIDChannelHeader
    data: np.ndarray | None = None

    def __post_init__(self):
        if self.data is None:
            self.data = np.empty(self.header.shape, self.header.dtype)

    def load_data_from_bytestream(self, iostream: BinaryIO) -> None:
        data_binary = iostream.read(self.header.sizeof())
        data_array_view = np.frombuffer(data_binary, self.header.dtype).reshape(
            self.data.shape
        )
        np.copyto(self.data, data_array_view)


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


@dataclass(frozen=True)
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
                channel_header = NIDChannelHeader(
                    self._header[f"DataSet-{group_idx}:{channel_idx}"]
                )
                channels.append(NIDChannel(channel_idx, group_idx, channel_header))

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
        groups = header.get_groups()
        for grp in groups:
            for chnl in grp.channels:
                chnl.load_data_from_bytestream(memorymap)

# %%
groups[0]

# %%
1838848 - 1826992
# %%
