import functools

import comet_ml
import optuna
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from geoinr.models import RLoss


def train(f: torch.nn.Module, dataset, opt: dict):
    dataloader = DataLoader(dataset, batch_size=1)

    for iteration, batch in enumerate(tqdm(dataloader)):
        break


def train_on_batch(f: torch.nn.Module, batch, opt: dict):
    f.train()

    return


def val(f: torch.nn.Module, dataset, opt: dict):
    f.eval()
    return


def objective(trial: optuna.Trial, dataset, opt: dict, device="cuda"):
    return metric


class Tune:
    def __init__(self):
        return

    objective = functools.partial(objective, dataset=dataset, device="cuda")

    study = optuna.create_study(
        study_name="geoinr_test",
        direction="minimize",
        # pruner=optuna.pruners.ThresholdPruner(upper=0.1, n_warmup_steps=100),
        storage="sqlite:///inr.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1)
