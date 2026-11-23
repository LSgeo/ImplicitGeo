"""Run GeoINR (load data, train, infer, plot) as one Nextflow-friendly command.

This replicates the flow of the repo-root ``__main__.py``, with every option
exposed through a config instead of hard-coded or read from the environment.

Run it from inside a Nextflow task (or anywhere ``geoinr`` is importable)::

    python3 -m geoinr.nextflow_main \\
        --config hparams.json \\
        --data-path WA_ANGD_airborne_ASCII.nc \\
        --outdir . \\
        --train.initial_lr 1e-4

Option precedence, lowest to highest:

    dataclass defaults  <  --config JSON file  <  dotted CLI overrides

The JSON file is grouped like the dataclasses below
(``data``, ``model``, ``train``, ``infer``, ``runtime``); a CLI override is
``--<group>.<option> <value>``. Unknown groups or options are errors, so typos
fail immediately instead of being silently ignored. ``--print-config`` prints
the resolved config and exits without importing torch.

Files written to ``--outdir`` (the process also changes into it, because
``Exp.log_figure`` writes PNGs to the current directory):

    resolved_config.json   merged config; can be fed straight back to --config
    metrics.json           split sizes, final validation MSE, grid stats, timings
    model.pt               state_dict (when runtime.run_training is true)
    figures/*.png          sample locations, preview grid, grid comparison
    INR: Dataset *.png     progress figures written by Exp.log_figure

Infrastructure (Slurm account, queue, GPUs, nodes) deliberately does not live
here; keep it in ``nextflow.config``.
"""

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Config
#
# Only the standard library is imported above. Everything that needs torch is
# imported lazily inside run(), so config parsing, --help and --print-config
# stay fast and work on a login node without the container.
#
# Defaults reproduce the values hard-coded in __main__.py. Names match the
# keys of the original ``train_opts`` dict wherever one existed.
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    variable: str = "cscba"  # netCDF variable to fit
    train_frac: float = 0.05  # random_split fractions; must sum to 1
    val_frac: float = 0.02
    test_frac: float = 0.93


@dataclass
class ModelConfig:
    nonlinearity: str = "siren"  # siren | sinusoidal | wire | wire2d | wire3d
    hidden_layers: int = 3
    hidden_features: int = 256
    # Siren(first_omega_0=30, hidden_omega_0=30.0) are the class defaults the
    # original script relied on; exposed here so they can be tuned.
    siren_first_omega_0: float = 30.0
    siren_hidden_omega_0: float = 30.0
    wire_omega_0: float = 3.0  # "Best for real data?" (1 for synthetic)
    wire_sigma: float = 3.0


@dataclass
class TrainConfig:
    total_epochs: int = 100
    batch_size: int = 2_048  # -1 means the whole training split
    initial_lr: float = 2e-4
    scheduler: str = "oclr"  # oclr (OneCycleLR) | mslr (MultiStepLR)
    weight_floss: float = 0.0  # Laplacian (physics) loss weight
    weight_rloss: float = 0.0  # frequency-bias regularisation loss weight
    rloss_Sigma: float = 2e-2  # noqa: N815  (norm width is 2)
    # Original: n_samples = int(0.01 * len(dataset)). Its comment says 10%.
    rloss_n_samples_frac: float = 0.01


@dataclass
class InferConfig:
    shortname: str = "gswa"  # prefix for figure file names
    # plt_sample_locs picks its colour scale from "alt" or "mag" in this label.
    value_label: str = "Bouger (maGl)"
    altitude: float = 1.0  # only used in the grid comparison file name
    z_mod: float = -1.0  # normalised z of the full grid slice
    preview_size: int = 2000  # preview grid is preview_size x preview_size
    grid_res: int = 1000  # full grid is grid_res x grid_res
    chunksize: int = 1_024_000  # points per inference batch
    vmin: float = -1000.0  # colour limits of the grid comparison plot
    vmax: float = 1000.0
    dpi: int = 96


@dataclass
class RuntimeConfig:
    device: str = "cuda"  # ROCm builds of torch also use "cuda"
    seed: int = 0
    tf32: bool = True
    use_amp: bool = False
    pin_memory: bool = True
    run_training: bool = True  # false: skip training and load a checkpoint


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_GROUPS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "infer": InferConfig,
    "runtime": RuntimeConfig,
}

_NONLINEARITIES = ("siren", "sinusoidal", "wire", "wire2d", "wire3d")
_SIREN_NAMES = ("siren", "sinusoidal")


class ConfigError(ValueError):
    """Raised for any problem with the JSON file or the command-line overrides."""


def _coerce(value, target: type, where: str):
    """Convert a JSON value or CLI string to ``target`` or raise ConfigError."""
    if target is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "1", "yes", "y", "on"):
                return True
            if text in ("false", "0", "no", "n", "off"):
                return False
        raise ConfigError(f"{where}: expected true/false, got {value!r}")

    if target is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        try:
            number = float(value)  # accepts 1e5 and "1e5"
        except (TypeError, ValueError):
            number = None
        if number is not None and math.isfinite(number) and number.is_integer():
            return int(number)
        raise ConfigError(f"{where}: expected an integer, got {value!r}")

    if target is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
        if not math.isfinite(number):
            raise ConfigError(f"{where}: expected a finite number, got {value!r}")
        return number

    if target is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where}: expected a string, got {value!r}")

    raise TypeError(f"unsupported config type {target!r} at {where}")


def _set_option(cfg: Config, group: str, key: str, raw, source: str) -> None:
    section = getattr(cfg, group)
    known = {f.name: f for f in dataclasses.fields(section)}
    if key not in known:
        raise ConfigError(
            f"{source}: unknown option '{group}.{key}'. "
            f"Valid '{group}' options: {', '.join(known)}"
        )
    setattr(section, key, _coerce(raw, known[key].type, f"{source}: {group}.{key}"))


def _merge_json(cfg: Config, raw, source: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: the top level must be a JSON object")
    for group, body in raw.items():
        if group not in _GROUPS:
            raise ConfigError(
                f"{source}: unknown group '{group}'. Valid groups: {', '.join(_GROUPS)}"
            )
        if not isinstance(body, dict):
            raise ConfigError(f"{source}: '{group}' must be a JSON object")
        for key, value in body.items():
            _set_option(cfg, group, key, value, source)


def _validate(cfg: Config) -> None:
    d, m, t, i, r = cfg.data, cfg.model, cfg.train, cfg.infer, cfg.runtime
    problems = []

    fracs = (d.train_frac, d.val_frac, d.test_frac)
    if any(f <= 0 for f in fracs):
        problems.append("data: split fractions must all be > 0")
    elif not math.isclose(sum(fracs), 1.0) or sum(fracs) > 1.0:
        # torch's random_split rejects float fractions summing to more than 1.
        problems.append(f"data: split fractions must sum to 1 (got {sum(fracs)!r})")

    if m.nonlinearity not in _NONLINEARITIES:
        problems.append(
            f"model.nonlinearity must be one of {_NONLINEARITIES}, got {m.nonlinearity!r}"
        )
    if m.hidden_layers < 1 or m.hidden_features < 1:
        problems.append("model: hidden_layers and hidden_features must be >= 1")

    if t.total_epochs < 1:
        problems.append("train.total_epochs must be >= 1")
    if t.batch_size != -1 and t.batch_size < 1:
        problems.append("train.batch_size must be >= 1, or -1 for the whole split")
    if t.initial_lr <= 0:
        problems.append("train.initial_lr must be > 0")
    if "oclr" not in t.scheduler and "mslr" not in t.scheduler:
        problems.append(
            f"train.scheduler must be 'oclr' or 'mslr', got {t.scheduler!r}"
        )
    if t.weight_floss < 0 or t.weight_rloss < 0:
        problems.append("train: weight_floss and weight_rloss must be >= 0")
    if t.rloss_Sigma <= 0:
        problems.append("train.rloss_Sigma must be > 0")
    if not 0 < t.rloss_n_samples_frac <= 1:
        problems.append("train.rloss_n_samples_frac must be in (0, 1]")

    label = i.value_label.lower()
    if "mag" not in label and "alt" not in label:
        problems.append(
            "infer.value_label must contain 'mag' or 'alt' (any case): "
            "geoinr.utils.plt_sample_locs chooses its colour scale from those "
            "substrings and fails without one"
        )
    if min(i.preview_size, i.grid_res, i.chunksize, i.dpi) < 1:
        problems.append("infer: preview_size, grid_res, chunksize and dpi must be >= 1")

    if not r.device:
        problems.append("runtime.device must not be empty")

    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


def resolve_config(config_path: Optional[Path], overrides: dict) -> Config:
    """Defaults, then the JSON file, then dotted CLI overrides; then validate."""
    cfg = Config()
    if config_path is not None:
        try:
            raw = json.loads(Path(config_path).read_text())
        except (OSError, json.JSONDecodeError) as err:
            raise ConfigError(
                f"could not read config file {config_path}: {err}"
            ) from err
        _merge_json(cfg, raw, str(config_path))
    for dotted, value in overrides.items():
        group, key = dotted.split(".", 1)
        _set_option(cfg, group, key, value, "command line")
    _validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    section = parser.add_argument_group(
        "hyperparameter overrides",
        "One flag per config option, as --<group>.<option> VALUE. "
        "Booleans take true/false. Only flags you pass override the config.",
    )
    defaults = Config()
    for group, cls in _GROUPS.items():
        for f in dataclasses.fields(cls):
            default = getattr(getattr(defaults, group), f.name)
            section.add_argument(
                f"--{group}.{f.name}",
                dest=f"{group}.{f.name}",
                default=argparse.SUPPRESS,
                metavar=f.type.__name__.upper(),
                help=f"default: {default!r}".replace("%", "%%"),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m geoinr.nextflow_main",
        description="Train, infer and plot a GeoINR model from a config.",
        epilog="Precedence: defaults < --config JSON < dotted CLI overrides.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, help="JSON file of hyperparameters")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="netCDF dataset to fit (required unless --print-config)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="directory for all outputs; the process changes into it (default: .)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="state_dict to load when runtime.run_training is false "
        "(default: <outdir>/model.pt)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the resolved config as JSON and exit",
    )
    _add_override_arguments(parser)
    return parser


# --------------------------------------------------------------------------- #
# The run itself (mirrors __main__.py)
# --------------------------------------------------------------------------- #


def _build_model(cfg: Config):
    import torch

    from geoinr.models import Siren
    from geoinr.models import get_INR as wire

    m = cfg.model
    if m.nonlinearity in _SIREN_NAMES:
        model = Siren(
            in_features=3,
            out_features=1,
            hidden_features=m.hidden_features,
            hidden_layers=m.hidden_layers,
            outermost_linear=True,
            first_omega_0=m.siren_first_omega_0,
            hidden_omega_0=m.siren_hidden_omega_0,
        )
    else:
        # NOTE: the original __main__.py tests `in ["wire2d, wire3d"]` (one
        # string), so it could never reach this branch. Dispatch is written
        # correctly here; this path has not been exercised on this branch.
        model = wire(
            nonlin=m.nonlinearity,  # "wire" | "wire2d" | "wire3d"
            in_features=3,
            out_features=1,
            hidden_features=m.hidden_features,
            hidden_layers=m.hidden_layers,
            first_omega_0=m.wire_omega_0,
            hidden_omega_0=m.wire_omega_0,
            scale=m.wire_sigma,
        )

    # get_INR returns (rather than raises) NotImplementedError for unknown names.
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"Could not build a model for {m.nonlinearity!r}: {model!r}")
    return model


def _train_opts(cfg: Config, n_points: int) -> dict:
    """The dict geoinr.trainval.Exp expects; keys match the original train_opts."""
    t, m, r = cfg.train, cfg.model, cfg.runtime
    return {
        "total_epochs": t.total_epochs,
        "batch_size": t.batch_size,
        "initial_lr": t.initial_lr,
        "scheduler": t.scheduler,
        "nonlinearity": m.nonlinearity,
        "hidden_layers": m.hidden_layers,
        "hidden_features": m.hidden_features,
        "weight_floss": t.weight_floss,
        "weight_rloss": t.weight_rloss,
        "rloss_Sigma": t.rloss_Sigma,
        "n_samples": int(t.rloss_n_samples_frac * n_points),
        "wire_omega_0": m.wire_omega_0,
        "wire_sigma": m.wire_sigma,
        "device": r.device,
        "use_amp": r.use_amp,
    }


def _mean_val_mse(model, val_dataloader, device: str) -> float:
    """Mean of per-batch MSE, matching Exp.val_epoch's averaging."""
    import torch

    model.eval()
    model.return_coords = False  # Exp.val_epoch can leave this truthy
    losses = []
    with torch.no_grad():
        for xyz, u in val_dataloader:
            pred = model(xyz.to(device, non_blocking=True))
            target = u.to(device, non_blocking=True)
            losses.append(torch.nn.functional.mse_loss(pred, target).item())
    return sum(losses) / len(losses)


def run(cfg: Config, data_path: Path, outdir: Path, checkpoint: Optional[Path]) -> None:
    t_start = time.perf_counter()
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)  # Exp.log_figure writes PNGs relative to the cwd
    Path("resolved_config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2) + "\n"
    )

    # Headless nodes: choose the backend before geoinr imports pyplot.
    import matplotlib

    matplotlib.use("Agg")
    import colorcet as cc
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import random_split

    import geoinr.trainval
    from geoinr.batcheddatasets import BatchingDataloader, NCDataset
    from geoinr.utils import plt_inr, plt_sample_locs, query_inr, query_inr_batched

    d, t, i, r = cfg.data, cfg.train, cfg.infer, cfg.runtime
    device = r.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device is 'cuda' but torch.cuda.is_available() is False. "
            "Check the GPU allocation and that the ROCm container is in use."
        )

    torch.manual_seed(r.seed)
    random.seed(r.seed)
    torch.backends.cuda.matmul.allow_tf32 = r.tf32
    torch.backends.cudnn.allow_tf32 = r.tf32

    fig_out = Path("figures")
    fig_out.mkdir(exist_ok=True, parents=True)
    model_path = Path("model.pt")

    # ---- Data ------------------------------------------------------------- #
    print(data_path)
    dataset = NCDataset(data_path, variable=d.variable)

    # Relies on the global seed set above, as the original did. Under DDP every
    # rank must produce the identical split (revisit in the distributed stage).
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [d.train_frac, d.val_frac, d.test_frac]
    )

    print(f"{len(train_dataset) = }, {len(val_dataset) = }")
    print(
        f"Percent train split: "
        f"{(len(train_dataset) / (len(val_dataset) + len(train_dataset))) * 100:0.1f}%"
    )
    print(f"Dropped {dataset.invalid_points} invalid points")
    print(f"Mean Alt: {dataset.unnormalise(dataset.xyz[:, 2], 'up').mean().item():.2f}")
    print(f"Std Alt: {dataset.unnormalise(dataset.xyz[:, 2], 'up').std().item():.2f}")

    fig = plt_sample_locs(
        dataset,
        indices=train_dataset.indices,
        unnormalise_fn=dataset.unnormalise,
        label=i.value_label,
        u=dataset.u[train_dataset.indices],
    )
    fig.savefig(fig_out / f"{i.shortname}_sample_locs.png", dpi=100)
    plt.close(fig)

    train_opts = _train_opts(cfg, n_points=dataset.u.shape[0])
    train_dataloader = BatchingDataloader(
        train_dataset, batch_size=t.batch_size, pin_memory=r.pin_memory
    )
    val_dataloader = BatchingDataloader(
        val_dataset, batch_size=t.batch_size, pin_memory=r.pin_memory
    )

    # ---- Model and training ---------------------------------------------- #
    model = _build_model(cfg)
    train_seconds = None
    if r.run_training:
        model = model.to(device, non_blocking=True)
        exp = geoinr.trainval.Exp(
            model,
            {"train": train_dataloader, "val": val_dataloader},
            train_opts,
        )
        t_train = time.perf_counter()
        inr = exp.train_inr()
        exp.exp.end()
        train_seconds = time.perf_counter() - t_train
        torch.save(inr.state_dict(), model_path)
        checkpoint = model_path

    # Load (also round-trips the checkpoint that was just written)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    # ---- Inference -------------------------------------------------------- #
    model.eval()
    inr = model.to(device, non_blocking=True)
    final_val_mse = _mean_val_mse(inr, val_dataloader, device)
    print(f"Final validation MSE: {final_val_mse:.6g}")

    alt = train_dataloader.xyz[0][:, :, 2].mean()
    u = query_inr(inr, (i.preview_size, i.preview_size, 1), z_mod=alt)

    fig = plt_inr(
        u.squeeze(),
        extent=train_dataloader.extent,
        suffix=f" {alt:0.3} m",
        ax_args=dict(cmap=cc.cm.CET_L1),
        figsize=(5, 5),
        dpi=i.dpi,
    )
    fig.savefig(fig_out / f"{i.shortname}.png", dpi=i.dpi)
    plt.close(fig)

    shape = (i.grid_res, i.grid_res, 1)
    b = [*dataset.var_ranges["e"], *dataset.var_ranges["n"]]
    u = query_inr_batched(
        inr,
        shape=shape,
        chunksize=i.chunksize,
        x_vec=dataset.normalise(torch.linspace(b[0], b[1], steps=shape[0]), "e"),
        y_vec=dataset.normalise(torch.linspace(b[2], b[3], steps=shape[1]), "n"),
        z_mod=i.z_mod,
    )
    u = dataset.unnormalise(u, "u")
    print(f"{u.mean() = :.2f}, {u.std() = :.2f}")
    print(f"{u.min() = :.2f}, {u.max() = :.2f}")

    train_u = dataset.unnormalise(dataset.u[train_dataset.indices], "u")
    print(f"{train_u.mean():.2f} {train_u.std():.2f}")
    print(f"{train_u.min():.2f} {train_u.max():.2f}")

    torch.cuda.empty_cache()  # clear anything left over from training/loading

    for z_slice in np.expand_dims(u, 2).transpose(2, 0, 1):  # iterate *zxy* slices
        fig = plt_inr(
            z_slice,
            extent=b,
            suffix=f" {i.shortname}",
            ax_args=dict(cmap=cc.cm.CET_L1, vmin=i.vmin, vmax=i.vmax),
            _vmin=-500,
            _vmax=500,
            residual=False,
            figsize=(10, 10),
            dpi=100,
        )
        out = fig_out / f"{i.shortname}_grid_comparison_{int(i.altitude)}m.png"
        fig.savefig(out, dpi=i.dpi)
        print(out.absolute())
        plt.close(fig)

    # ---- Metrics ---------------------------------------------------------- #
    metrics = {
        "run": {
            "data_path": str(data_path),
            "checkpoint": str(checkpoint),
            "trained": r.run_training,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "hip_version": getattr(torch.version, "hip", None),
        },
        "data": {
            "n_points": len(dataset),
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_test": len(test_dataset),
            "invalid_points": int(dataset.invalid_points),
        },
        "results": {
            "final_val_mse": final_val_mse,
            "grid_mean": float(u.mean()),
            "grid_std": float(u.std()),
            "grid_min": float(u.min()),
            "grid_max": float(u.max()),
        },
        "timing": {
            "train_seconds": train_seconds,
            "total_seconds": time.perf_counter() - t_start,
        },
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # keep Slurm / Nextflow logs current
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if "." in k}

    try:
        cfg = resolve_config(args.config, overrides)
    except ConfigError as err:
        parser.error(str(err))  # exits with status 2

    if args.print_config:
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return 0

    if args.data_path is None:
        parser.error("--data-path is required (unless --print-config is used)")
    # Resolve every path before run() changes directory.
    data_path = args.data_path.resolve()
    if not data_path.is_file():
        parser.error(f"--data-path does not exist: {data_path}")

    outdir = args.outdir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else outdir / "model.pt"
    if not cfg.runtime.run_training and not checkpoint.is_file():
        parser.error(
            f"runtime.run_training is false but no checkpoint exists at {checkpoint}"
        )

    run(cfg, data_path, outdir, checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
