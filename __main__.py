"""Implicit function toolbox for geophysics.

If we can fit a point data to an Implicit Neural Network, we can operate directly on the Neural Network representation.
"""

import os
import random
from pathlib import Path

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Maybe not on WSL
import colorcet as cc
import numpy as np
import pyproj
import torch
from torch.utils.data import random_split

# import rasterio as rio
import geoinr.trainval
from geoinr.batcheddatasets import BatchingDataloader, NCDataset
from geoinr.models import Siren
from geoinr.models import get_INR as wire
from geoinr.utils import plt_inr, plt_sample_locs, query_inr, query_inr_batched

torch.manual_seed(seed := 0)
random.seed(seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda"

fig_out = Path("figures")
fig_out.mkdir(exist_ok=True, parents=True)
model_outpath = "model.pt"
working_dir = Path(os.environ["WORKDIR"])

# Berigora Dataset
shortname = "gswa"
netcdf_path = working_dir / "WA_ANGD_airborne_ASCII.nc"
print(netcdf_path.absolute())
dataset = NCDataset(netcdf_path, variable="cscba")
altitude = 1  # Read dataset.ncd description
cell_size = 10
res = 1000
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.5, 0.2, 0.3])

print(f"{len(train_dataset) = }, {len(val_dataset) = }")
print(
    f"Percent train split: {(len(train_dataset) / (len(val_dataset) + len(train_dataset))) * 100:0.1f}%"
)
print(f"Dropped {dataset.invalid_points} invalid points")

print(f"Mean Alt: {dataset.unnormalise(dataset.xyz[:, 2], 'up').mean().item():.2f}")
print(f"Std Alt: {dataset.unnormalise(dataset.xyz[:, 2], 'up').std().item():.2f}")

# Plot training data
fig = plt_sample_locs(
    dataset,
    indices=train_dataset.indices,
    # label="mag (nT)",
    unnormalise_fn=dataset.unnormalise,
    label="Bouger (maGl)",
    # u=altitude,
    u=dataset.u[train_dataset.indices],
)
fig.savefig(fig_out / f"{shortname}_sample_locs.png", dpi=100)


# Create Model
train_opts = dict(
    total_epochs=100_000,
    batch_size=2_048_000,  # 1M for 12 GB
    initial_lr=2e-4,
    scheduler="oclr",
    nonlinearity="siren",
    hidden_layers=5,
    hidden_features=512,
    weight_floss=0,  # 10,  # .01, #4e-4,  # 1e-3, #1e-4,  # 1e-4,
    weight_rloss=0,  # 1e-3, #1e-4,  # 1e-4,
    rloss_Sigma=2e-2,  # norm width is 2
    n_samples=int(0.01 * dataset.u.shape[0]),  # 10% samples for RLoss
    wire_omega_0=3,  # Best for real data?
    wire_sigma=3,
    # wire_omega_0=1,   # Best for synthetic data
    # wire_sigma=1,
    device=device,
    use_amp=False,
)

train_dataloader = BatchingDataloader(
    train_dataset, batch_size=train_opts["batch_size"], pin_memory=True
)

val_dataloader = BatchingDataloader(
    val_dataset, batch_size=train_opts["batch_size"], pin_memory=True
)

if train_opts["nonlinearity"] in ["sinusoidal", "siren"]:
    comet_tags = ["SIREN"]
    model: torch.nn.Module = Siren(
        in_features=3,
        out_features=1,
        hidden_features=train_opts["hidden_features"],
        hidden_layers=train_opts["hidden_layers"],
        outermost_linear=True,
    )

elif train_opts["nonlinearity"] in ["wire2d, wire3d"]:
    comet_tags = ["WIRE"]
    model: torch.nn.Module = wire(
        nonlin=train_opts["nonlinearity"],  # "wire2d" | "wire3d"
        in_features=3,
        out_features=1,
        hidden_features=train_opts["hidden_features"],
        hidden_layers=train_opts["hidden_layers"],
        first_omega_0=train_opts["wire_omega_0"],  # first Frequency of sinusoid
        hidden_omega_0=train_opts["wire_omega_0"],  # hidden Frequency of sinusoid
        scale=train_opts["wire_sigma"],  # Sigma of Guassian
    )
else:
    raise ValueError("Need to load a model.")

if not isinstance(model, torch.nn.Module):
    raise ValueError


# Train
if True:
    model = model.to(device, non_blocking=True)

    exp = geoinr.trainval.Exp(
        model,
        {"train": train_dataloader, "val": val_dataloader},
        train_opts,
    )
    inr = exp.train_inr()
    exp.exp.end()

    # Save
    torch.save(inr.state_dict(), model_outpath)

# Load
model.load_state_dict(torch.load(model_outpath, weights_only=True))

# Inference
model.eval()
inr = model.to(device, non_blocking=True)

alt = train_dataloader.xyz[0][:, :, 2].mean()
u = query_inr(inr, (2000, 2000, 1), z_mod=alt)

# Shortcut plot everything
fig = plt_inr(
    u.squeeze(),
    extent=train_dataloader.extent,
    suffix=f" {alt:0.3} m",
    ax_args=dict(cmap=cc.cm.CET_L1),
    figsize=(5, 5),
    dpi=1000,
)
fig.savefig(fig_out / f"{shortname}.png", dpi=1000)

# Advanced plotting
normalise_fn = dataset.normalise
unnormalise_fn = dataset.unnormalise

extent = [
    unnormalise_fn(dataset.extent[0], "e"),
    unnormalise_fn(dataset.extent[1], "e"),
    unnormalise_fn(dataset.extent[2], "n"),
    unnormalise_fn(dataset.extent[3], "n"),
]
shape = (res, res, 1)
b = [*dataset.var_ranges["e"], *dataset.var_ranges["n"]]
u = query_inr_batched(
    inr,
    shape=shape,
    chunksize=1_024_000,
    x_vec=normalise_fn(torch.linspace(b[0], b[1], steps=shape[0]), "e"),
    y_vec=normalise_fn(torch.linspace(b[2], b[3], steps=shape[1]), "n"),
    z_mod=-1,
)
u = unnormalise_fn(u, "u")
print(f"{u.mean() = :.2f}, {u.std() = :.2f}")
print(f"{u.min() = :.2f}, {u.max() = :.2f}")

print(
    f"{unnormalise_fn(dataset.u[train_dataset.indices], 'u').mean():.2f} "
    f"{unnormalise_fn(dataset.u[train_dataset.indices], 'u').std():.2f}"
)
print(
    f"{unnormalise_fn(dataset.u[train_dataset.indices], 'u').min():.2f} "
    f"{unnormalise_fn(dataset.u[train_dataset.indices], 'u').max():.2f}"
)

# Hopefully clear anything leftover after training, loading, etc.
torch.cuda.empty_cache()

# Visualise result
for i, z_slice in enumerate(
    np.expand_dims(u, 2).transpose(2, 0, 1)
):  # Iter through *zxy* slices
    # altitude = unnormalise_fn(z_vec[i], "up")
    fig = plt_inr(
        z_slice,
        extent=b,
        suffix=f" {shortname}",
        ax_args=dict(
            cmap=cc.cm.CET_L1,
            vmin=-1000,
            vmax=1000,
        ),
        _vmin=-500,
        _vmax=500,
        residual=False,
        figsize=(10, 10),
        dpi=100,
    )

    fig.savefig(fig_out / f"{shortname}_grid_comparison_{int(altitude)}m.png", dpi=1000)
    print((fig_out / f"{shortname}_grid_comparison_{int(altitude)}m.png").absolute())
