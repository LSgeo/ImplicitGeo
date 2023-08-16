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
        self.easting = "e"
        self.northing = "n"
        self.upward = "up"
        self.xyz = None
        self.u = None
        self.var_ranges = {
            "e": None,
            "n": None,
            "up": None,
            "u": None,
        }
        self.invalid_points = 0
        self.extent = (-1, 1, -1, 1, -1, 1)

    def __len__(self):
        return len(self.u)

    def _normalise(self, inp, var: str, a=-1, b=1) -> torch.Tensor:
        """Min-Max Normalise between upper and lower -1 and 1
        This private method records the original ranges
        """
        self.a = a
        self.b = b
        self.var_ranges[var] = (np.min(inp), np.max(inp))

        return (b - a) * ((inp - np.min(inp)) / (np.max(inp) - np.min(inp))) + a

    def normalise(self, inp, var: str, a=-1, b=1) -> torch.Tensor:
        """Normalise inputs to the range of the training data set in _normalise"""
        _min = min(self.var_ranges[var])
        _max = max(self.var_ranges[var])

        return (b - a) * ((inp - _min) / (_max - _min)) + a

    def unnormalise(self, inp, var: str) -> torch.Tensor:
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
        if not matched:
            raise ValueError(
                f"Couldn't find a match in {possible_names} for {self.ncd.variables.keys()}"
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
        self.easting = self.find_variable_name(["easting", "x"])
        self.northing = self.find_variable_name(["northing", "y"])
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
