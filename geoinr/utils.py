import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# import tifffile
import torch

from geoinr.batcheddatasets import construct_xyz

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


def query_inr(inr, shape=(200, 200, 10), **kwargs) -> np.ndarray:
    """Generate coordinates to query trained INR model
    Suitable for small shapes, otherwise see query_inr_batched
    kwargs define coord query and are passed to construct_xyz
    """
    xyz = construct_xyz(shape, **kwargs).unsqueeze(0)
    xyz = xyz.to(device=device, non_blocking=True)
    inr.return_coords = False
    u = inr(xyz)
    return u.detach().cpu().view(shape).rot90().squeeze().numpy()


def generate_inr_batches(inr, shape, chunksize, **kwargs) -> torch.Tensor:
    """Generate coordinates to query trained INR model
    kwargs define coord query and are passed to construct_xyz

    #TODO: if this gets slow, preallocate the storage, fix below
    # full_uxyz = torch.empty(shape[0] * shape[1] * shape[2])
    # full_uxyz[int(i*bu.shape[1]):int((i+1)*bu.shape[1])] = bu.view([0,:,0]

    """
    xyz = construct_xyz(shape, **kwargs).unsqueeze(0)
    # for z in xyz[:, :, 2]:
    for batch in torch.split(xyz, chunksize, dim=1):
        batch = batch.to(device=device, non_blocking=True)
        inr.return_coords = False
        u = inr(batch)

        yield u.detach().cpu()


def query_inr_batched(
    inr, shape=(200, 200, 10), chunksize=256_000, **kwargs
) -> np.ndarray:
    full_u = []
    for b_u in generate_inr_batches(inr, shape, chunksize, **kwargs):
        full_u.append(b_u)

    return torch.hstack(full_u).view(shape).rot90().squeeze().numpy()


def plt_inr(
    u,
    extent,
    ax_args,
    suffix=None,
    _vmin=None,
    _vmax=None,
    gt_grid=None,
    residual=False,
    cropping=(0, -1, 0, -1),
    **kwargs,
):
    """Plot a default INR model output comparison"""
    if gt_grid is not None:
        if residual:
            fig, [ax0, ax1, ax2] = plt.subplots(1, 3, constrained_layout=True, **kwargs)
        else:
            fig, [ax0, ax1] = plt.subplots(1, 2, constrained_layout=True, **kwargs)
        orientation = "horizontal"
    else:
        fig, ax1 = plt.subplots(1, 1, constrained_layout=True, **kwargs)
        orientation = "vertical"

    w, e, s, n = cropping  # TODO These do not correspond to west, east...

    # fig.suptitle(f"INR Comparison")  # , Altitude = {altitude:0.2f}")

    ax1.set_title(f"INR{suffix}")
    im1 = ax1.imshow(u[:, :][w:e, s:n], extent=extent, **ax_args)
    # plt.colorbar(im1, ax=ax1, orientation="horizontal", label="nT")
    ax1.set_xlabel("Easting")
    # ax1.set_ylabel("Northing")
    ax1.ticklabel_format(useOffset=False)

    if gt_grid is not None:
        ax0.set_title("Reference grid")
        ax0.imshow(gt_grid[w:e, s:n], extent=extent, **ax_args)
        ax0.set_xlabel("Easting")
        ax0.set_ylabel("Northing")
        ax0.ticklabel_format(useOffset=False)
        # Share INR grid cmap
        plt.colorbar(im1, ax=[ax0, ax1], orientation=orientation, label="nT")
    else:
        plt.colorbar(im1, ax=ax1, orientation=orientation, label="nT")

    if residual:
        ax2.set_title(
            f"Residual (RMS: {rms(gt_grid[w:e, s:n], u[:, :][w:e, s:n]):0.2f} nT)"
            # f"Residual (PSNR: {psnr(gt_grid[w:e, s:n], u[:, :][w:e, s:n]):0.2f})"
        )
        if not _vmax:
            std = u.std()
            _vmax = 2 * std
            _vmin = -2 * std

        imdiff = ax2.imshow(
            gt_grid[w:e, s:n] - u[:, :][w:e, s:n],
            vmin=_vmin,
            vmax=_vmax,
            cmap=cc.cm.CET_D7,
            extent=extent,
        )
        imdiff.cmap.set_under("k")
        plt.colorbar(
            imdiff, ax=ax2, orientation="horizontal", label=r"$\Delta$nT", aspect=9
        )
        ax2.set_xlabel("Easting")
        ax2.ticklabel_format(useOffset=False)
        # ax2.set_ylabel("Northing")

    return fig


def psnr(grid1, grid2):
    mse = np.mean((grid1.astype(np.float64) - grid2.astype(np.float64)) ** 2)
    return 10 * np.log10((grid1.max() - grid1.min()) ** 2 / mse)


def rms(grid1, grid2):
    return np.sqrt(np.mean((grid1 - grid2) ** 2))


def plt_sample_locs(
    dset, indices=None, label=None, unnormalise_fn=None, gtt=None, u=None, ax3d=False
):
    # from matplotlib.colors import TwoSlopeNorm

    if unnormalise_fn is not None:
        x = unnormalise_fn(dset.xyz[:, 0], "e").numpy()
        y = unnormalise_fn(dset.xyz[:, 1], "n").numpy()
        z = unnormalise_fn(dset.xyz[:, 2], "up").numpy()
    else:
        x = dset.xyz[:, 0]
        y = dset.xyz[:, 1]
        z = dset.xyz[:, 2]

    if indices:
        x = x[indices]
        y = y[indices]
        z = z[indices]

    if "alt" in label.lower():
        clr = z
        cmap = cc.cm.CET_D1
        vmin = u - 20
        vmax = u + 20

    elif "mag" in label.lower():
        if u is None:
            raise ValueError("Must also specify u argument")
        clr = unnormalise_fn(u, "u").numpy()
        cmap = cc.cm.CET_L1
        vmin = None
        vmax = None

    fig = plt.figure(
        figsize=(e_size("1.5"), e_size("1.5")*0.9),
        layout="constrained",
    )
    if not ax3d:
        ax = fig.add_subplot()
        clrs_alt = ax.scatter(
            x,
            y,
            c=clr,
            s=0.5,
            alpha=0.6,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    else:
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=30, azim=250)
        ax.set_zlabel("Altitude")
        clrs_alt = ax.scatter(
            x, y, z, c=clr, s=0.05, alpha=0.8, cmap=cmap, vmin=vmin, vmax=vmax
        )

    # norm=TwoSlopeNorm(40),
    ax.set_xlabel(f"Easting {chr(176)}")
    ax.set_ylabel(f"Northing {chr(176)}")
    plt.axis("equal")
    plt.colorbar(clrs_alt, label=label)  # , orientation="horizontal")

    if gtt is not None:
        plt.imshow(gtt, cmap=cc.cm.CET_L1)

    return fig


def plt_kwargs(
    ax_args,
    suptitle=None,
    shape=None,
    unit=chr(176),  # default axis unit is degree sign
    **kwargs,
) -> plt.Figure:
    """Custom plot function bespoke to my figure layouts and content"""

    label = kwargs.pop("label", None)
    figsize = kwargs.pop("figsize", (7.48, 7.48 * 2 / 3))
    std = kwargs.pop("std", None)
    transect_coords = kwargs.pop("transect_coords", None)
    shape = shape or (1, len(kwargs.keys()))
    if "xlim" in kwargs or "ylim" in kwargs:
        lims = {"xlim": kwargs.pop("xlim", None), "ylim": kwargs.pop("ylim", None)}
    else:
        lims = {}

    if len(kwargs.keys()) > (shape[0] * shape[1]):
        raise ValueError("Insufficient shape for keyword args")

    styles = ["k-", "g-", "b-", "r-"]

    fig, axs = plt.subplots(
        *shape,
        layout="constrained",
        figsize=figsize,
        sharex=False,  # True,
        sharey=False,  # True,
        subplot_kw=lims,
    )
    axs = np.array(axs)

    fig.suptitle(suptitle)

    rax = []
    cax = []
    for i, (ax, (name, im)) in enumerate(zip(axs.ravel(), kwargs.items())):
        if name.lower() in ["off", "none"]:
            ax.axis("off")
            continue

        ax.set_title(name)
        ax.ticklabel_format(useOffset=False, style="plain")
        # ax.tick_params(axis="x", labelrotation=-90)
        # ax.tick_params(axis="y", labelrotation=-90)

        if "Residual" in name:
            _vmin = ax_args.pop("vmin")
            _vmax = ax_args.pop("vmax")
            std2 = std
            rim = ax.imshow(
                im, **{**ax_args, "cmap": cc.cm.CET_D1, "vmin": -std2, "vmax": std2}
            )
            rax.append(ax)
        else:
            cim = ax.imshow(im, **ax_args)
            # if i > shape[1]:  # Enforce first row cbar on first row
            if True:  # Enforce first row cbar on first row
                cax.append(ax)

        if transect_coords and i < len(styles):
            style = styles[i]
            x0, x1, y0, y1 = transect_coords
            ax.plot([x0, x1], [y0, y1], style)
            ax.annotate(
                "A",
                (x0, y0),
                c=style[0],
                backgroundcolor=("gray", 0.5),
                xytext=(0.25, 0.8),
                textcoords="offset fontsize",
            )
            ax.annotate(
                "A'",
                (x1, y1),
                c=style[0],
                backgroundcolor=("gray", 0.5),
                xytext=(-1.25, 0.8),
                textcoords="offset fontsize",
            )

        if i == 4:
            # Align plot to gridspec - if ax on low row should share upper bar
            tmp_bar = plt.colorbar(cim, ax=ax, orientation="horizontal")
            tmp_bar.remove()

        if "Residual" in name:
            ax_args["vmin"] = _vmin
            ax_args["vmax"] = _vmax

        if i in [0, shape[1]]:
            ax.set_ylabel(f"Northing {unit}")  # (m)")
        if shape[1] == 2 or shape[0] == 1 or i in range(shape[1], shape[0] * shape[1]):
            ax.set_xlabel(f"Easting {unit}")  # (m)")

    plt.colorbar(cim, ax=cax, orientation="horizontal", label=label)
    if "Residual" in name:
        plt.colorbar(rim, ax=rax, orientation="horizontal", label=label, aspect=10)

    return fig


def transect(ims: np.ndarray, gtt: np.ndarray, indice: int) -> matplotlib.figure.Figure:
    """Simple reliable transect, one whole row or column"""

    styles = ["g-", "b-", "r-"]
    # styles = ["g-.", "b:", "r--"]
    labels = ["INR", "Eq. Sources", "Bicubic"]
    ann_args = dict(
        xycoords="axes fraction", backgroundcolor=("gray", 0.2), annotation_clip=False
    )

    # plt.title(f"Synthetic transect")
    plt.figure(figsize=(e_size("1.5"), 3), constrained_layout=True)
    plt.plot(
        np.linspace(0, ims[1].shape[1], gtt.shape[1], endpoint=True),
        gtt[indice * 4, :],
        "k-",
        drawstyle="steps-mid",
        label="GT",
    )
    for im, style, label in zip(np.array(ims[1:])[:, indice, :], styles, labels):
        plt.plot(im, style, drawstyle="steps-mid", label=label)

    plt.grid(True)
    plt.annotate("A", xy=(0.01, 0.02), **ann_args)
    plt.annotate("A'", xy=(0.970, 0.02), **ann_args)
    plt.xlabel("Easting")
    plt.xlim(0, 50)
    plt.xticks(
        np.linspace(0, ims[1].shape[1], 11), labels=np.linspace(0, 4000, 11, dtype=int)
    )
    plt.ylabel("TMI (nT)")
    plt.legend()

    return plt.gcf()


def e_size(s) -> float:
    """Calculate inch for pyplot from elsevier figure widths
    https://beta.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing

    s: named size in ["minimal", "single", "double"/"full"], or size in mm
    """
    if isinstance(s, (int, float)):
        mm = s
    elif isinstance(s, str):
        if s.lower() in ["minimal"]:
            mm = 30
        elif s == "1" or s.lower() in ["single"]:
            mm = 90
        elif s == "1.5":
            mm = 140
        elif s == "2" or s.lower() in ["double", "full"]:
            mm = 190
        else:
            raise ValueError(
                "Unsupported target size: Use minimal, single, double/full"
            )
    else:
        raise ValueError(f"{s=}, {mm=}")
    return mm / 25.4


def crop_gtt(gtt, clip_bounds):
    import rasterio
    from shapely.geometry import Polygon

    w, e, s, n = clip_bounds
    clip = Polygon(((w, n), (e, n), (e, s), (w, s)))
    return rasterio.mask.mask(gtt, [clip], crop=True)


def load_nc_grid(grid_path, crop=False, nan_val=-99999):
    """Load grid data from a netCDF, such as those provided by GA on the
    dapds00 thredds server.
    """

    from pathlib import Path
    import rasterio
    import rioxarray
    import xarray

    grid_path = Path(grid_path)
    xds = xarray.open_dataset(grid_path)
    xds.Band1.rio.to_raster(f"{grid_path.stem}.tif")

    gtt = rasterio.open(f"{grid_path.stem}.tif")
    b = [b for b in gtt.bounds]
    gtt_extent = [b[0], b[2], b[1], b[3]]

    if crop:
        gtt = crop_gtt(gtt, crop)[0][0]
        gtt_extent = crop
    else:
        gtt = gtt.read(1)

    gtt[gtt == nan_val] = float("nan")

    return gtt, gtt_extent
