from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset


rng = np.random.default_rng()

if torch.cuda.is_available():
    device = "cuda"


def merge_z_slices(dir_path: str, idx: int = 0, n: int = 50):
    """Merge multiple Noddy forward models into a single 3D array
    For example, create a 200x200x200 volume of synthetic measurements.
    Args:
        dir_path: Path to directory to merge all .mag (and .grv)
        idx: Which model dir index to load
        n: number of files to include, i.e. number of slices
    """

    dir_path = Path(dir_path)

    mag_files = sorted(list(dir_path.glob("*/"))[idx].glob("*.mag"))[:n]  # sel idx dir
    # grv_files = sorted(list(dir_path.glob("*"))[idx].glob("*.grv"))[:n]

    # get x y extent from header
    # _, xlen, ylen, _ = mag_files[0].read_text().splitlines()[3].split()
    # zlen = len(mag_files)

    return np.stack(
        [np.genfromtxt(f, dtype=np.float32, skip_header=8) for f in mag_files], axis=-1
    )


class INRDataset(Dataset):
    """Base class for Implicit Neural Representation Dataset"""

    def __init__(self):
        super().__init__()
        self.coords = None
        self.cells = None
        self.pre_norms = {"x": None, "y": None, "z": None, "u": None}

    def __len__(self):
        return 1  # This is an Implicit Function to overfit 1 sample

    def subsample(self, s: int = 1, mode="xyz_step"):
        """Subsample the dataset to reduce RAM usage"""
        if s == 1:
            return
        if mode == "vol_step":
            self.u = self.u[::s, ::s, ::s]
        if mode == "xyz_step":
            self.coords = self.coords[:, ::s, :]
            self.cells = self.cells[::s, :]
        if mode == "xyz_random_n":
            idcs = rng.choice(
                np.arange(len(self.cells)),
                len(self.cells) // s,
                replace=False,
                shuffle=False,
            )
            self.coords = self.coords[:, idcs, :]
            self.cells = self.cells[idcs, :]

    def _normalise(self, inp, var: str, a=-1, b=1):
        """Min-Max Normalise between upper and lower -1 and 1"""
        self.a = a
        self.b = b
        self.pre_norms[var] = {"min": inp.min(), "max": inp.max()}

        return (b - a) * ((inp - inp.min()) / (inp.max() - inp.min())) + a

    def unnormalise(self, inp, var: str):
        _min = self.pre_norms[var]["min"]
        _max = self.pre_norms[var]["max"]

        return (inp - self.a) * ((_max - _min) / (self.b - self.a)) + _min

    def split_train_val(self, train_pct: float = 0.85):
        """Split the dataset into train and validation sets"""

        num_train = int(self.cells.shape[0] * train_pct)
        idcs = torch.randperm(self.cells.shape[0], device="cpu")

        self.train_coords = self.coords[:, idcs[:num_train], :]
        self.train_cells = self.cells[idcs[:num_train], :]

        self.val_coords = self.coords[:, idcs[num_train:], :]
        self.val_cells = self.cells[idcs[num_train:], :]

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError("This dataset should always have length 1")
        if self.cells is None or self.coords is None:
            raise ValueError("Dataset has not been init")

        return {
            "train_coords": self.train_coords,
            "train_cells": self.train_cells,
            "val_coords": self.val_coords,
            "val_cells": self.val_cells,
        }


class PointData3D(INRDataset):
    """Construct a dataset for SIREN comprising 3D point coordinates
    and point values
    """

    def __init__(self, xyz_dir, idx, subsample=1):
        super().__init__()
        self.xyz_dir = xyz_dir
        self.u = merge_z_slices(self.xyz_dir, idx=idx)
        self._subsample(subsample, mode="3d")
        self.og_shape = self.u.shape
        self.coords = self.get_mgrid()
        self.cells = torch.from_numpy(self.u).contiguous().view(-1, 1)
        self.coords = self._normalise(self.coords)
        self.cells = self._normalise(self.cells)

    def get_mgrid(self):
        tensors = tuple((torch.linspace(-1, 1, steps=s) for s in self.u.shape))
        coords = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        coords = coords.reshape(-1, len(tensors)).unsqueeze(0)
        return coords


class NCDataset(INRDataset):
    def __init__(self, netcdf_path: Path, variable: str):
        super().__init__()

        self.netcdf_path = Path(netcdf_path)
        self.variable = variable
        self._load_nc()
        # self.split_train_val()

    def _load_nc(self):
        self.ncd = netCDF4.Dataset(self.netcdf_path, "r")
        self.coords = tuple(
            (
                torch.from_numpy(self._normalise(self.ncd.variables["x"][:], "x")),
                torch.from_numpy(self._normalise(self.ncd.variables["y"][:], "y")),
                torch.from_numpy(
                    self._normalise(self.ncd.variables["altitude"][:], "z")
                ),
            )
        )

        self.coords = (
            torch.stack(self.coords, dim=-1)
            .reshape(-1, 3)
            .unsqueeze(0)
            .to(torch.float32)
        )
        try:
            self.cells = torch.from_numpy(
                self._normalise(self.ncd.variables[self.variable][:], "u")
            )
        except KeyError:
            raise KeyError(f"Variable not found in netCDF file, options are {self.ncd.variables.keys()}")

        self.cells = self.cells.contiguous().view(-1, 1)

    def plot_variables(self, variables: list, **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (10, 5)), dpi=kwargs.get("dpi", 160))

        for i, variable in enumerate(variables):
            v = self.ncd.variables[variable][:]
            c = ["k", "r", "g", "b"][i]
            plt.scatter(np.arange(len(v)), v, s=kwargs.get("s", 0.05), c=c, label=f"{variable}")
            plt.axhline(np.median(v), c="k", label=f"{variable}_median")
            # plt.title(f"norm mean {((1 - -1) * ((i - i.min()) / (i.max() - i.min())) - 1).mean()}")
            # plt.text(0.5, 60, f"Median Alt: {np.median(i):0.2f}")

        plt.legend()
        plt.xlabel("index")
        plt.ylabel(variable)
        plt.show()
