import functools

import colorcet as cc
import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from tqdm.auto import tqdm

from geoinr.models import RLoss
from geoinr.utils import query_inr, plt_inr


class Exp:
    def __init__(self, f: torch.nn.Module, dataloaders: dict, opt: dict):
        self.f = f
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.opt = opt
        self.step = 0
        self.device = opt["device"]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.opt["use_amp"])

    def train_inr(self) -> torch.nn.Module:
        self.init_comet()
        trial = None

        self.optim = torch.optim.Adam(
            lr=self.opt["initial_lr"], params=self.f.parameters()
        )
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.opt["total_epochs"],
            max_lr=self.opt["initial_lr"],
        )

        self.cri_mse = torch.nn.MSELoss()
        self.cri_r = RLoss(Sigma=self.opt["rloss_Sigma"], device=self.opt["device"])

        self.step = 0
        for epoch in tqdm(
            range(self.opt["total_epochs"]), unit="epoch", desc="Training"
        ):
            self.exp.set_epoch(epoch)
            self.train_epoch()

            if (epoch + 1) % 100 == 0:
                val_metric = self.val_epoch()
            if (epoch + 1) % 250 == 0:
                self.log_figure()

            if trial is not None:
                trial.report(val_metric, self.step)
                if trial.should_prune():
                    self.exp.add_tag("Pruned")
                    self.exp.end()
                    raise optuna.TrialPruned()

        self.exp.end()
        print(f"Total steps: {self.step}")

        return self.f

    def train_epoch(self):
        self.f.train()
        self.f.return_coords = True

        for i, batch in enumerate(self.train_dataloader):
            self.exp.set_step(self.step)
            if i % 200 == 0:
                self.exp.log_parameter("lr", self.sched.get_last_lr())

            # Send the data to the model and predict
            train_u = batch[1].to(self.device, non_blocking=True)
            train_xyz = batch[0].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.opt["device"], enabled=self.opt["use_amp"]):
                pred_u, xyz = self.f(train_xyz)

                # Calculate Loss
                loss_mse = self.cri_mse(pred_u, train_u)
                if self.opt["weight_rloss"] > 0:
                    xbar = torch.rand_like(train_xyz[:, : self.opt["n_samples"], :])
                    loss_r = self.cri_r(self.f, xbar, self.opt["n_samples"])
                    loss_total = loss_mse + self.opt["weight_rloss"] * loss_r
                else:
                    loss_total = loss_mse

            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            # loss_total.backward()
            self.optim.zero_grad(set_to_none=True)
            # self.optim.step()
            self.sched.step()

            # Log metrics
            if i % 250 == 0:
                self.exp.log_metric("Train Loss Total", loss_total.item())
                self.exp.log_metric("Train Loss MSE", loss_mse.item())
                if self.opt["weight_rloss"] > 0:
                    self.exp.log_metric("Train Loss Regularisation", loss_r.item())

            self.step += 1

    @torch.no_grad()
    def val_epoch(self):
        self.f.eval()
        self.f.return_coords = False

        avg_metric = []
        for vi, d in enumerate(self.val_dataloader):
            val_xyz = d[0].to(self.device, non_blocking=True)
            val_u = d[1].to(self.device, non_blocking=True)

            pred_u = self.f(val_xyz)

            avg_metric.append(self.cri_mse(pred_u, val_u).item())

        avg_metric = np.array(avg_metric).mean()
        self.exp.log_metric("Val Loss MSE", avg_metric)

        return avg_metric

    def init_comet(self):
        if "sinusoidal" in self.opt["nonlinearity"]:
            comet_tags = ["SIREN"]
        elif "wire" in self.opt["nonlinearity"]:
            comet_tags = ["WIRE"]

        self.exp = comet_ml.Experiment(disabled=False)
        self.exp.add_tags(comet_tags)
        self.exp.log_code("geoinr/models.py")
        self.exp.log_code("geoinr/datasets.py")
        self.exp.log_parameter("dataset", self.train_dataloader.name)
        self.exp.log_parameters(self.opt)

    @torch.no_grad()
    def log_figure(self):
        u = query_inr(
            self.f, (200, 200, 1), z_mod=self.train_dataloader.normalise(39, "up")
        )

        fig = plt_inr(
            u.squeeze(),
            altitude=39,
            extent=self.train_dataloader.extent,
            ax_args=dict(cmap=cc.cm.CET_L1),
            figsize=(5, 5),
            dpi=100,
        )
        self.exp.log_figure("INR: Selected height", figure=fig, step=self.step)
        plt.close()


# def objective(trial: optuna.Trial, dataset, opt: dict, device="cuda"):
#     return None


# class Tune:
#     def __init__(self):
#         return

#     dataset = None
#     objective = functools.partial(objective, dataset=dataset, device="cuda")

#     study = optuna.create_study(
#         study_name="geoinr_test",
#         direction="minimize",
#         # pruner=optuna.pruners.ThresholdPruner(upper=0.1, n_warmup_steps=100),
#         storage="sqlite:///inr.db",
#         load_if_exists=True,
#     )
#     study.optimize(objective, n_trials=1)


# if tune := 0:
#     def objective(trial: optuna.Trial, dataset, device):
#         tune_opts = {
#             **train_opts,
#             **dict(
#                 initial_lr=trial.suggest_float("initial_lr", 4e-5, 4e-5, log=True),
#                 hidden_layers=trial.suggest_int("hidden_layers", 3, 3),
#                 omega_0=trial.suggest_float("omega_0", 10, 30, step=10),
#                 sigma=trial.suggest_float("sigma", 10, 30, step=10),
#             ),
#         }

#     objective = functools.partial(objective, dataset=dataset, device=device)

#     study = optuna.create_study(
#         study_name="geoinr_test",
#         direction="minimize",
#         # pruner=optuna.pruners.ThresholdPruner(upper=0.1, n_warmup_steps=100),
#         storage="sqlite:///inr.db",
#         load_if_exists=True,
#     )
#     study.optimize(objective, n_trials=1)
