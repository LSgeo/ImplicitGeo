import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tifffile
import torch

from geoinr.datasets import construct_xyz

rng = np.random.default_rng()

if torch.cuda.is_available():
    device = "cuda"


def plt_cntr(u, **kwargs):
    im = plt.imshow(u, origin="lower", cmap=cc.cm.CET_L1, **kwargs)
    cset = plt.contour(
        u,
        np.linspace(u.min(), u.max(), 10),
        linewidths=1,
        cmap=cc.cm.CET_R2,
        **kwargs,
    )
    plt.clabel(cset, inline=True, fmt="%1.1f", fontsize=8)
    plt.colorbar(im, orientation="horizontal", label="nT")


def plt_3d(u, ori="z", levels=25, step=10, **kwargs):
    x, y, z = np.meshgrid(
        np.arange(u.shape[0]),
        np.arange(u.shape[1]),
        np.arange(u.shape[2]),
    )

    kw = dict(
        vmin=u.min(),
        vmax=u.max(),
        cmap=kwargs.get("cmap", cc.cm.CET_L20),
        # norm=colors.PowerNorm(gamma=0.5),
        norm=colors.AsinhNorm(linear_width=kwargs.get("linear_width", 1)),
        levels=np.linspace(u.min(), u.max(), levels),
        alpha=kwargs.get("alpha", 1 / step),
    )

    # Create a figure with 3D ax
    if kwargs.get("fig"):
        fig = kwargs.get("fig")
        ax = kwargs.get("ax")
    else:
        fig = plt.figure(figsize=(5, 4), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")

    fig.suptitle(kwargs.get("title", "Potential field volume"))

    # Plot contour surfaces
    _ = ax.contourf(
        x[0, :, :],
        u[0, :, :],
        z[0, :, :],
        zdir="y",
        offset=0,
        **{**kw, "alpha": 0.7},
    )
    _ = ax.contourf(
        u[:, -1, :],
        y[:, -1, :],
        z[:, -1, :],
        zdir="x",
        offset=x.max(),
        **{**kw, "alpha": 0.7},
    )
    C = ax.contourf(
        x[:, :, 0],
        y[:, :, 0],
        u[:, :, 0],
        zdir="z",
        offset=z.min(),
        **{**kw, "alpha": 0.7},
    )

    if ori == "z":  # Contour flat in Z
        for i in np.arange(z.min() + step, z.max() + 1, step):
            _ = ax.contour(x[:, :, 0], y[:, :, 0], u[:, :, i], zdir="z", offset=i, **kw)

    elif ori == "x":  # Contour flat in X
        for i in np.arange(x.min() + step, x.max() + 1, step):
            _ = ax.contour(
                u[:, i, :], y[:, -1, :], z[:, -1, :], zdir="x", offset=i, **kw
            )

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    zmin, zmax = z.min(), z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    edges_kw = dict(color="0.4", linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    ax.set(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        xticks=[int(xmax * i) for i in [0, 0.33, 0.66, 1]],
        yticks=[int(ymax * i) for i in [0, 0.33, 0.66, 1]],
        zticks=[int(zmax * i) for i in [0, 0.33, 0.66, 1]],
    )

    ax.view_init(kwargs.get("elev", 45), kwargs.get("azim", 45), 0)
    ax.set_box_aspect(None, zoom=1)

    fig.colorbar(C, ax=ax, fraction=0.04, pad=0.2, label="nT")

    if kwargs.get("fig"):
        return

    plt.show()


def query_inr(inr, shape=(100, 100, 1), **kwargs):
    """Generate coordinates to query trained INR model
    Suitable for small shapes, otherwise see query_inr_batched
    kwargs define coord query and are passed to construct_xyz
    """
    xyz = construct_xyz(shape, **kwargs).unsqueeze(0)
    xyz = xyz.to(device=device, non_blocking=True, dtype=torch.float32)

    u, _ = inr(xyz)
    u = u.detach().cpu().view(shape)
    u = u.rot90()

    return u.squeeze().numpy()


def generate_inr_batches(inr, shape=(200, 200, 10), chunksize=256_000, **kwargs):
    """Generate coordinates to query trained INR model
    kwargs define coord query and are passed to construct_xyz

    #TODO: if this gets slow, preallocate the storage, fix below
    # full_uxyz = torch.empty(shape[0] * shape[1] * shape[2])
    # full_uxyz[int(i*bu.shape[1]):int((i+1)*bu.shape[1])] = bu.view([0,:,0]

    """
    xyz = construct_xyz(shape, **kwargs).unsqueeze(0)
    # for z in xyz[:, :, 2]:
    for batch in torch.split(xyz, chunksize, dim=1):
        batch = batch.to(device=device, non_blocking=True, dtype=torch.float32)
        u, _ = inr(batch)

        yield u.detach().cpu().squeeze()


def query_inr_batched(inr, shape=(200, 200, 10), chunksize=256_000, **kwargs):
    full_u = []
    for b_u in generate_inr_batches(
        inr,
        shape=shape,
        chunksize=chunksize,
        z_vec=kwargs.get("z_vec"),
    ):
        full_u.append(b_u)

    full_u = torch.hstack(full_u).view(shape).rot90().permute(2, 0, 1).numpy()

    return full_u


def plt_inr(
    u,
    altitude,
    extent,
    ax_args,
    z_slice=0,
    _vmin=None,
    _vmax=None,
    gt_grid=None,
    cropping=(0, -1),
    **kwargs,
):
    """Plot a default INR model output comparison"""
    if gt_grid is not None:
        fig, [ax0, ax1, ax2] = plt.subplots(1, 3, constrained_layout=True, **kwargs)
    else:
        fig, ax1 = plt.subplots(1, 1, constrained_layout=True, **kwargs)

    fig.suptitle(f"INR Comparison, Altitude = {altitude:0.2f}")

    ax1.set_title("Implicit Neural Representation")
    im1 = ax1.imshow(u[:, :], extent=extent, **ax_args)
    plt.colorbar(im1, ax=ax1, orientation="horizontal")

    if gt_grid is not None:
        ax0.set_title("GT Grid from GA GADDS")
        ax0.imshow(gt_grid, **ax_args)
        plt.colorbar(im1, ax=ax0, orientation="horizontal")

        c0, c1 = cropping
        ax2.set_title("Residuals GT - INR")
        imdiff = ax2.imshow(
            gt_grid[c0:c1, c0:c1] - u[:, :][c0:c1, c0:c1],
            vmin=_vmin,
            vmax=_vmax,
            cmap=cc.cm.CET_D7,
            extent=extent,
        )
        plt.colorbar(imdiff, ax=ax2, orientation="horizontal")
    # else:
    #     ax0.axis("off")
    #     ax2.axis("off")

    return fig
