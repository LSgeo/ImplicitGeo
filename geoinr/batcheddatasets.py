from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset


rng = np.random.default_rng()

if torch.cuda.is_available():
    device = "cuda"


class INRDataset(Dataset):
    """Base class for Implicit Neural Representation Dataset"""

    def __init__(self):
        super().__init__()
        self.xyz = None
        self.u = None
        self.var_ranges = {"x": None, "y": None, "z": None, "u": None}
        self.extent = (-1, 1, -1, 1, -1, 1)

    def __len__(self):
        return len(self.u)

    def _normalise(self, inp, var: str, a=-1, b=1):
        """Min-Max Normalise between upper and lower -1 and 1
        This private method records the original ranges
        """
        self.a = a
        self.b = b
        self.var_ranges[var] = (np.min(inp), np.max(inp))

        return (b - a) * ((inp - np.min(inp)) / (np.max(inp) - np.min(inp))) + a

    def normalise(self, inp, var: str, a=-1, b=1):
        """Normalise inputs to the range of the training data set in _normalise"""
        _min = min(self.var_ranges[var])
        _max = max(self.var_ranges[var])

        return (b - a) * ((inp - _min) / (_max - _min)) + a

    def unnormalise(self, inp, var: str):
        _min = min(self.var_ranges[var])
        _max = max(self.var_ranges[var])

        return (inp - self.a) * ((_max - _min) / (self.b - self.a)) + _min

    def subsample_wesnbt(self, extent: tuple):
        """Normalised coordsys (W, E, S, N, lowest z, highest z)"""
        self.extent = extent
        idcs = (
            (self.xyz[:, 0] >= extent[0])
            & (self.xyz[:, 0] <= extent[1])
            & (self.xyz[:, 1] >= extent[2])
            & (self.xyz[:, 1] <= extent[3])
            & (self.xyz[:, 2] >= extent[4])
            & (self.xyz[:, 2] <= extent[5])
        )

        self.xyz = self.xyz[idcs]
        self.u = self.u[idcs]

        return self

class NCDataset(INRDataset):
    def __init__(self, file_path: Path, variable: str):
        super().__init__()

        self.file_path = Path(file_path)
        self.variable = variable
        self._load_nc()

    def _load_nc(self):
        self.ncd = netCDF4.Dataset(self.file_path, "r")
        self.xyz = tuple(
            (
                torch.from_numpy(self._normalise(self.ncd.variables["x"][:], "x")),
                torch.from_numpy(self._normalise(self.ncd.variables["y"][:], "y")),
                torch.from_numpy(
                    self._normalise(self.ncd.variables["altitude"][:], "z")
                ),
            )
        )

        self.xyz = torch.stack(self.xyz, dim=-1).reshape(-1, 3).to(torch.float32)
        try:
            self.u = torch.from_numpy(
                self._normalise(self.ncd.variables[self.variable][:], "u")
            )
        except KeyError:
            raise KeyError(
                f"Variable not found in netCDF file, options are {self.ncd.variables.keys()}"
            )

        self.u = self.u.contiguous().view(-1, 1)

    def __getitem__(self, idx):
        """This should not be used for batched training.
        (i.e. Don't use torch.DataLoader with this dataset - it will be slow!)
        """
        return {"xyz": self.xyz[idx], "u": self.u[idx]}


class BatchingDataloader:
    def __init__(self, dataset: torch.utils.data.Subset, batch_size: int, pin_memory=False, **kwargs):
        """Take Subset dataset and split into batches.
        Replaces torch.Dataloader
        """
        if kwargs:  # Quick swap with normal dataloader
            print(f"Yeeting unused kwargs {kwargs} into the void")

        self.dataset = dataset[:]  # Get all subset data
        self.pin_memory = pin_memory
        if batch_size == -1:
            self.batch_size = len(self.dataset["u"])
        else:
            self.batch_size = batch_size
        self.steps_per_epoch = np.ceil(len(self.dataset["u"]) / self.batch_size)

        self.prepare_batch_tensors()

    def prepare_batch_tensors(self):
        xyz = self.dataset["xyz"].unsqueeze(0)
        u = self.dataset["u"].unsqueeze(0)

        if self.pin_memory:
            self.xyz = torch.split(xyz.pin_memory(), self.batch_size, dim=1)
            self.u = torch.split(u.pin_memory(), self.batch_size, dim=1)
        else:
            self.xyz = torch.split(xyz, self.batch_size, dim=1)
            self.u = torch.split(u, self.batch_size, dim=1)

    def __len__(self):
        """Handy to access easily"""
        return int(self.steps_per_epoch)

    def __iter__(self):
        yield from zip(self.xyz, self.u)
