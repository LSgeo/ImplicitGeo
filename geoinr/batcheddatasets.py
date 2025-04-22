from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

import natsort


rng = np.random.default_rng()

if torch.cuda.is_available():
    device = "cuda"


def construct_xyz(
    shape, x_r=1, y_r=1, z_r=0, xy_mod=1, z_mod=0, **kwargs
) -> torch.Tensor:
    """Construct a coordinate space, perhaps to regularise an INR to.
    If you are going to do that:
    Args:
        shape: 3D h,w,c shape, which determines the resolution of the mesh.
        {x|y|z}_r should be the same as the normalised training space, i.e. -1 to 1.
        {x|y|z}_mod allows you to scale the coordinate axis.

        We treat z different - we specify a "middle" slice value, and range around it
    """

    if any(kw in kwargs for kw in ["zvec", "zmod"]):
        raise NotImplementedError("Don't forget the underscore!")

    x_vec = kwargs.get("x_vec", torch.linspace(-x_r, x_r, steps=shape[0]) * xy_mod)
    y_vec = kwargs.get("y_vec", torch.linspace(-y_r, y_r, steps=shape[1]) * xy_mod)
    z_vec = kwargs.get(
        "z_vec", torch.linspace(z_mod - z_r, z_mod + z_r, steps=shape[2])
    )

    xyz = torch.cartesian_prod(*tuple((x_vec, y_vec, z_vec)))

    if shape[-1] == 1:  # return mid value if only 1 slice
        try:
            xyz[:, -1] = torch.Tensor([1.0]) * (z_mod)  # TODO convert z_mod sanely
        except:
            xyz[:, -1] = z_mod
    if shape[-1] == 2:  # return mid value if only 1 slice
        # xyz[::2, -1] = z_mod # This works but is stupid.
        raise ValueError("Z shape should be either 1, or more than 2.")

    return xyz


def merge_z_slices(dir_path: str, n: int = None):  # , n: int = 50):
    """Merge multiple Noddy forward models into a single 3D array
    For example, create a 200x200x200 volume of synthetic measurements.
    Args:
        dir_path: Path to directory to merge all .mag (and .grv)
        n: number of files to include, i.e. number of slices
    """

    dir_path = Path(dir_path)
    mag_files = natsort.natsorted(list(dir_path.glob("*.mag*")))[:n]  # sel idx dir
    # grv_files = sorted(list(dir_path.glob("*"))[idx].glob("*.grv"))[:n]

    # get x y extent from header
    # _, xlen, ylen, _ = mag_files[0].read_text().splitlines()[3].split()
    # zlen = len(mag_files)

    return np.stack(
        [
            # np.genfromtxt(f, dtype=np.float32, skip_header=8)
            np.ascontiguousarray(np.loadtxt(f, skiprows=8, dtype=np.float32))
            for f in mag_files
        ],
        axis=-1,
    )


class INRDataset(Dataset):
    """Base class for Implicit Neural Representation Dataset"""

    def __init__(self):
        super().__init__()
        self.easting = "e"
        self.northing = "n"
        self.upward = "up"
        self.xyz = None
        self.u = None
        self.var_ranges = {"e": None, "n": None, "up": None, "u": None}
        self.invalid_points = 0
        self.extent = (-1, 1, -1, 1, -1, 1)

    def __len__(self):
        return len(self.u)

    def _normalise(self, inp, var: str, a=-1, b=1):
        """Min-Max Normalise between upper and lower -1 and 1
        This private method records the original ranges
        """
        self.a = a
        self.b = b
        if np.max(inp) == np.min(inp):
            if type(inp) is int or inp.dtype == "int":
                self.var_ranges[var] = (np.min(inp), np.max(inp))
                return np.zeros_like(inp)
            else:
                inp = inp.astype(np.float32)
                inp[0] += 0.000001
        self.var_ranges[var] = (np.min(inp), np.max(inp))
        return (b - a) * ((inp - np.min(inp)) / (np.max(inp) - np.min(inp))) + a

    def normalise(self, inp, var: str, a=-1, b=1):
        """Normalise inputs to the range of the training data set in _normalise"""
        _min = min(self.var_ranges[var])
        _max = max(self.var_ranges[var])
        if _max == _min:
            if type(inp) is int or type(inp) is float:
                return np.zeros_like(inp)
            else:
                inp[0] += 0.000001
                _max += 0.000001
        return (b - a) * ((inp - _min) / (_max - _min)) + a

    def unnormalise(self, inp, var: str) -> torch.Tensor:
        _min = min(self.var_ranges[var])
        _max = max(self.var_ranges[var])
        if _max == _min:
            try:
                inp[0] += 0.000001
            except:
                return np.zeros_like(inp)

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

    def __getitem__(self, idx) -> dict:
        """This should not be used for batched training.
        (i.e. Don't use torch.DataLoader with this dataset - it will be slow!)
        """
        return {"xyz": self.xyz[idx], "u": self.u[idx]}


class NCDataset(INRDataset):
    def __init__(self, file_path: Path, variable: str):
        super().__init__()

        self.file_path = Path(file_path)
        self.variable = variable
        self._load_nc()

    def find_variable_name(self, possible_names: list) -> str:
        try:
            matched = next(k for k in self.ncd.variables.keys() if k in possible_names)
        except StopIteration as v:
            raise ValueError(
                f"Couldn't find a match in {possible_names} for {self.ncd.variables.keys()}: {v}"
            )

        # While we are here, check for value variable name:
        if self.variable in possible_names and matched != self.variable:
            print("\n################")
            print(f"'{self.variable}' not found, using '{matched}' instead")
            print("################\n")

        return matched

    def _prepare_data(self):
        if "inf" not in self.ncd.geospatial_bounds:  # Hope for the best
            return (
                self.ncd.variables[self.easting][:],
                self.ncd.variables[self.northing][:],
                self.ncd.variables[self.upward][:],
                self.ncd.variables[self.variable][:],
            )
        else:  # Found potential NaNs in xyz
            valid_idcs = np.all(
                (
                    np.isfinite(self.ncd.variables[self.easting][:]),
                    np.isfinite(self.ncd.variables[self.northing][:]),
                    np.isfinite(self.ncd.variables[self.upward][:]),
                    ~self.ncd.variables[self.variable][:].mask,
                ),  # not masked is ok
                axis=0,
            )
            self.invalid_points = (
                self.ncd.variables[self.easting][:].size - valid_idcs.sum()
            )
            x = self.ncd.variables[self.easting][:][valid_idcs]
            y = self.ncd.variables[self.northing][:][valid_idcs]
            z = self.ncd.variables[self.upward][:][valid_idcs]
            u = self.ncd.variables[self.variable][:][valid_idcs]
            return x, y, z, u

    def _load_nc(self):
        self.ncd = netCDF4.Dataset(self.file_path, "r")
        self.easting = self.find_variable_name(["easting", "x", "longitude"])
        self.northing = self.find_variable_name(["northing", "y", "latitude"])
        self.upward = self.find_variable_name(["upward", "altitude"])
        self.variable = self.find_variable_name([self.variable, "mag_microLevelled"])

        x, y, z, u = self._prepare_data()

        self.xyz = tuple(
            (
                torch.from_numpy(self._normalise(x, "e")),
                torch.from_numpy(self._normalise(y, "n")),
                torch.from_numpy(self._normalise(z, "up")),
            )
        )
        self.xyz = torch.stack(self.xyz, dim=-1).reshape(-1, 3).to(torch.float32)

        self.u = torch.from_numpy(self._normalise(u, "u"))
        self.u = self.u.contiguous().view(-1, 1).to(torch.float32)

        # self.ncd.close()  # Only if needed to reduce CPU RAM

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


class CSVDataset(INRDataset):
    def __init__(self, file_path: Path, variable: str, usecols: list = None):
        super().__init__()

        self.file_path = Path(file_path)
        self.variable = variable
        self.usecols = usecols
        self._load_csv()

    def _override_variable(self, var: str, value: float):
        self.csv[var][0::2] = value + torch.randn(1)
        self.csv[var][1::2] = value - torch.randn(1)

    def _load_csv(self):
        self.csv = np.genfromtxt(
            self.file_path, delimiter=",", names=True, usecols=self.usecols
        )

        self._override_variable(self.usecols[2], 1500.0)

        self.xyz = tuple(
            (
                torch.from_numpy(self._normalise(self.csv[self.usecols[0]], "e")),
                torch.from_numpy(self._normalise(self.csv[self.usecols[1]], "n")),
                torch.from_numpy(self._normalise(self.csv[self.usecols[2]], "up")),
            )
        )

        self.xyz = torch.stack(self.xyz, dim=-1).reshape(-1, 3).to(torch.float32)
        try:
            self.u = torch.from_numpy(self._normalise(self.csv[self.usecols[3]], "u"))
        except KeyError:
            raise KeyError(
                f"Variable not found in csv file, options are in {self.csv.dtype}"
            )

        self.u = self.u.contiguous().view(-1, 1).to(torch.float32)


class NoddyDataset(INRDataset):
    """A dataset for INR on Noddy synthetic data
    Optionally limit the number of altitude slices to load
    """

    def __init__(
        self,
        file_path: Path,
        variable: str,
        file_limit: int = None,
        specific_id: int = None,
    ):
        super().__init__()

        self.file_path = Path(file_path)
        self.file_limit = file_limit
        self.variable = variable
        self._load_noddy(file_limit, specific_id)

    def get_mgrid(self, cs=20):
        """Make coord grid for noddy 20 m data"""
        tensors = [
            np.arange(s) * cs for s in self.u.unsqueeze(2).shape
        ]  # temp unsqueeze for 2D
        tensors[2] += 100  # Noddy data starts at z=100
        x, y, z = tensors
        xyz = tuple(
            (
                torch.from_numpy(self._normalise(x, "e")),
                torch.from_numpy(self._normalise(y, "n")),
                torch.from_numpy(self._normalise(z, "up")).to(torch.float64),
            )
        )
        xyz = (
            torch.stack(torch.meshgrid(*xyz, indexing="ij"), dim=-1)
            .reshape(-1, 3)
            .to(torch.float32)
        )
        return xyz

    def _load_noddy(self, file_limit, specific_id=None):
        if specific_id:  # temp slice for 2D
            self.u = merge_z_slices(self.file_path, n=file_limit)[:, :, specific_id]
        else:
            self.u = merge_z_slices(self.file_path, n=file_limit)

        self.u = torch.from_numpy(self._normalise(self.u, "u")).to(torch.float32)
        self.xyz = self.get_mgrid()
        self.u = self.u.contiguous().view(-1, 1)


class BatchingDataloader:
    def __init__(
        self,
        dataset: torch.utils.data.Subset,
        batch_size: int,
        pin_memory=False,
        **kwargs,
    ):
        """Take Subset dataset and split into batches.
        Replaces torch.Dataloader
        """
        if kwargs:  # Quick swap with normal dataloader
            print(f"Yeeting unused kwargs {kwargs} into the void")

        self.name = dataset.dataset.file_path.name
        self.extent = dataset.dataset.extent
        self.var_ranges = dataset.dataset.var_ranges
        self.normalise = dataset.dataset.normalise
        self.unnormalise = dataset.dataset.unnormalise
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


def get_agg_data():
    """Get Gravity Gradient data
    https://geophys-data.geoscience.nsw.gov.au/air_surveys/AIR_2003_gov_Broken_Hill_AGG_Mag_0350.zip
    """
    raise NotImplementedError()
    # import pandas as pd

    # names = "X Y LONGITUDE LATITUDE ALTITUDE FIDUCIAL RADAR ALT_DEM DEM TURBULENCE Err_NE Err_UV T_DD T_NE T_UV A_SJT_2p75_NE_ML A_SJT_2p75_UV_ML B_SJT_2p75_NE_ML B_SJT_2p75_UV_ML gD_FOURIER_2p75_400_ML_orig GDD_FOURIER_2p75_400_ML_orig GNE_FOURIER_2p75_400_ML_orig GUV_FOURIER_2p75_400_ML_orig DRAPESURFACE_FOURIER_FS gD_FOURIER_0p0_400_ML GDD_FOURIER_0p0_400_ML GNE_FOURIER_0p0_400_ML GUV_FOURIER_0p0_400_ML DRAPESURFACE_EQUIV gD_EQUIV_2p75_ML GDD_EQUIV_2p75_ML GNE_EQUIV_2p75_ML GUV_EQUIV_2p75_ML gD_FOURIER_2p75_400_ML GDD_FOURIER_2p75_400_ML GNE_FOURIER_2p75_400_ML GUV_FOURIER_2p75_400_ML GS_Fiducial GS_StrLine".split()
    # usecols = [
    #     "X Y DRAPESURFACE_EQUIV gD_EQUIV_2p75_ML GDD_EQUIV_2p75_ML GNE_EQUIV_2p75_ML GUV_EQUIV_2p75_ML"
    # ]

    # dat = pd.read_csv(
    #     "AIR_2003_gov_Broken_Hill_AGG_Mag_0350/Reprocessed_2011/Located_Data/AGG.dat",
    #     delim_whitespace=True,
    #     names=names,
    #     usecols=usecols,
    #     na_values="*",
    #     comment="/",
    # )
