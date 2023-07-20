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


def construct_xyz(shape, x_r=1, y_r=1, z_r=1, xy_mod=1, z_mod=0, **kwargs):
    """Construct a coordinate space, perhaps to regularise an INR to.
    If you are going to do that:
    Args:
        shape: 3D h,w,c shape, which determins the resolution of the mesh.
        {x|y|z}_r should be the same as the normalised training space, i.e. -1 to 1.
        {x|y|z}_mod allows you to scale the coordinate axis.

        We treat z different - we specify a "middle" slice value, and range around it
    """
    z_vec = kwargs.get("z_vec")
    if z_vec is None:
        z_vec = torch.linspace(z_mod - z_r, z_mod + z_r, steps=shape[2])

    xyz = torch.cartesian_prod(
        *tuple(
            (
                torch.linspace(-x_r, x_r, steps=shape[0]) * xy_mod,
                torch.linspace(-y_r, y_r, steps=shape[1]) * xy_mod,
                # torch.linspace(-z_r, z_r, steps=shape[2]) * z_mod,
                z_vec,
            )
        )
    )

    if shape[-1] == 1:  # return mid value if only 1 slice
        xyz[:, -1] = z_mod
    if shape[-1] == 2:  # return mid value if only 1 slice
        # xyz[::2, -1] = z_mod # This works but is stupid.
        raise ValueError("Z shape should be either 1, or more than 2.")

    return xyz


class INRDataset(Dataset):
    """Base class for Implicit Neural Representation Dataset"""

    def __init__(self):
        super().__init__()
        self.xyz = None
        self.u = None
        self.var_ranges = {"x": None, "y": None, "z": None, "u": None}

    def __len__(self):
        return 1  # This is an Implicit Function to overfit 1 sample

    def subsample(self, s: int = 1, mode="xyz_step"):
        """Subsample the dataset to reduce RAM usage"""
        if s == 1:
            return
        if mode == "vol_step":
            self.u = self.u[::s, ::s, ::s]
        if mode == "xyz_step":
            self.xyz = self.xyz[::s, :]
            self.u = self.u[::s, :]
        if mode == "xyz_random_n":
            idcs = rng.choice(
                np.arange(len(self.u)),
                len(self.u) // s,
                replace=False,
                shuffle=False,
            )
            self.xyz = self.xyz[idcs, :]
            self.u = self.u[idcs, :]

    def _normalise(self, inp, var: str, a=-1, b=1):
        """Min-Max Normalise between upper and lower -1 and 1
        This private method records the original ranges
        """
        self.a = a
        self.b = b
        self.var_ranges[var] = {"min": inp.min(), "max": inp.max()}

        return (b - a) * ((inp - inp.min()) / (inp.max() - inp.min())) + a

    def normalise(self, inp, var: str, a=-1, b=1):
        """Normalise inputs to the range of the training data set in _normalise"""
        _min = self.var_ranges[var]["min"]
        _max = self.var_ranges[var]["max"]

        return (b - a) * ((inp - _min) / (_max - _min)) + a

    def unnormalise(self, inp, var: str):
        _min = self.var_ranges[var]["min"]
        _max = self.var_ranges[var]["max"]

        return (inp - self.a) * ((_max - _min) / (self.b - self.a)) + _min

    def split_train_val(self, train_pct: float = 0.85):
        """Split the dataset into train and validation sets"""

        num_train = int(self.u.shape[0] * train_pct)
        idcs = torch.randperm(self.u.shape[0], device="cpu")

        self.train_xyz = self.xyz[idcs[:num_train], :]
        self.train_u = self.u[idcs[:num_train], :]

        self.val_xyz = self.xyz[idcs[num_train:], :]
        self.val_u = self.u[idcs[num_train:], :]

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError("This dataset should always have length 1")
        if self.u is None or self.xyz is None:
            raise ValueError("Dataset has not been init")

        return {
            "train_xyz": self.train_xyz,
            "train_u": self.train_u,
            "val_xyz": self.val_xyz,
            "val_u": self.val_u,
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
        self.xyz = self.get_mgrid()
        self.u = torch.from_numpy(self.u).contiguous().view(-1, 1)
        self.xyz = self._normalise(self.xyz, "xyz")
        raise NotImplementedError("x,y,z need to be normalised independently")
        self.u = self._normalise(self.u, "z")

    def get_mgrid(self):
        tensors = tuple((torch.linspace(-1, 1, steps=s) for s in self.u.shape))
        xyz = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        xyz = xyz.reshape(-1, len(tensors))
        return xyz


class NCDataset(INRDataset):
    def __init__(self, netcdf_path: Path, variable: str):
        super().__init__()

        self.netcdf_path = Path(netcdf_path)
        self.variable = variable
        self._load_nc()
        # self.split_train_val()

    def _load_nc(self):
        self.ncd = netCDF4.Dataset(self.netcdf_path, "r")
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

    def plot_variables(self, variables: list, **kwargs):
        plt.figure(figsize=kwargs.get("figsize", (10, 5)), dpi=kwargs.get("dpi", 160))

        for i, variable in enumerate(variables):
            v = self.ncd.variables[variable][:]
            c = ["k", "r", "g", "b"][i]
            plt.scatter(
                np.arange(len(v)), v, s=kwargs.get("s", 0.05), c=c, label=f"{variable}"
            )
            plt.axhline(np.mean(v), c="k", label=f"{variable}_mean")
            # plt.title(f"norm mean {((1 - -1) * ((i - i.min()) / (i.max() - i.min())) - 1).mean()}")
            # plt.text(0.5, 60, f"Median Alt: {np.median(i):0.2f}")

        plt.legend()
        plt.xlabel("index")
        plt.ylabel(variable)
        plt.show()
