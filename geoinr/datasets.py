from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import torch

from geoinr.batcheddatasets import INRDataset

rng = np.random.default_rng()

if torch.cuda.is_available():
    device = "cuda"


class INRDataset_nonbatched(INRDataset):
    """Base class for Implicit Neural Representation Dataset"""

    def __init__(self):
        super().__init__()

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

    def __getitem__(self, idx) -> dict:
        if idx > 0:
            raise IndexError("This dataset should always have length 1")
        if self.u is None or self.xyz is None:
            raise ValueError("Dataset has not been init")

        return {"xyz": self.xyz, "u": self.u}


class PointData3D(INRDataset_nonbatched):
    """Construct a dataset for SIREN comprising 3D point coordinates
    and point values
    """

    def __init__(self, xyz_dir, idx, subsample=1):
        super().__init__()
        self.xyz_dir = xyz_dir
        self.u = merge_z_slices(self.xyz_dir, idx=idx)
        self.subsample(subsample, mode="3d")
        self.og_shape = self.u.shape
        self.xyz = self.get_mgrid()
        self.u = torch.from_numpy(self.u).contiguous().view(-1, 1)
        self.xyz = self._normalise(self.xyz, "xyz")
        raise NotImplementedError("x,y,z need to be normalised independently")
        self.u = self._normalise(self.u, "z")



# class CustomDataloader:
#     def __init__(self, dataset, batch_size: int, pin_memory=False, **kwargs):
#         if kwargs:  # Quick swap with normal dataloader
#             print(f"Yeeting unused kwargs {kwargs} into the void")
#         self.dataset = dataset[0]  # This only works with BatchedINRdset
#         self.steps_per_epoch = len(dataset)
#         self.pin_memory = pin_memory
#         if batch_size == -1:
#             self.batch_size = len(dataset)
#         else:
#             self.batch_size = batch_size

#         self.prepare_batch_tensors()

#     def prepare_batch_tensors(self):
#         d = self.dataset

#         if self.pin_memory:
#             self.xyz = torch.split(d["xyz"].pin_memory(), self.batch_size)
#             self.u = torch.split(d["u"].pin_memory(), self.batch_size)
#         else:
#             self.xyz = torch.split(d["xyz"], self.batch_size)
#             self.u = torch.split(d["u"], self.batch_size)

#     def __iter__(self):
#         yield from zip(self.xyz, self.u)