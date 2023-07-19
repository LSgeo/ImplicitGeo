import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tifffile
import torch

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
        alpha=kwargs.get("alpha", 0.1),
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


def query_inr(
    unnormaliser,
    model,
    i=0,
    h=200,
    w=200,
    c=1,
    x_r=1,
    y_r=1,
    z_r=1,
    xy_mod=1,
    z_mod=1,
):
    """Generate coordinates to query trained INR model"""
    tensors = tuple(
        (
            torch.linspace(-x_r, x_r, steps=h) * xy_mod,
            torch.linspace(-y_r, y_r, steps=w) * xy_mod,
            torch.linspace(-z_r, -z_r, steps=c) * z_mod,
        )
    )

    un = unnormaliser
    extent = [
        un(-x_r * xy_mod, "x"),
        un(x_r * xy_mod, "x"),
        un(-y_r * xy_mod, "y"),
        un(y_r * xy_mod, "y"),
    ]
    height = un((torch.linspace(-z_r, z_r, steps=c) * z_mod)[i], "z")

    coords = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    coords = coords.reshape(-1, len(tensors)).unsqueeze(0).to(torch.float32)
    coords = coords.to("cuda", non_blocking=True)

    new_u, _ = model(coords)
    new_u = new_u.detach().cpu().view((h, w, c))
    new_u = unnormaliser(new_u, "u").rot90().numpy()

    return new_u, extent, height


def plt_inr(
    u,
    extent,
    height,
    ax_args,
    i=0,
    gt_tiff=None,
    **kwargs,
):
    """Plot a default INR model output comparison"""
    fig, [ax0, ax1] = plt.subplots(1, 2, constrained_layout=True, **kwargs)
    fig.suptitle(f"INR Model Output Comparison")

    ax1.set_title(f"INR, height = {height:0.2f}")
    im1 = ax1.imshow(u[:, :, i], extent=extent, **ax_args)
    plt.colorbar(im1, ax=ax1, orientation="horizontal")

    if gt_tiff is not None:
        ax0.set_title(f"GT Grid from GA GADDS")
        ax0.imshow(tifffile.imread(gt_tiff), **ax_args)
        plt.colorbar(im1, ax=ax0, orientation="horizontal")
    else:
        ax0.axis("off")

    return fig
