import time
from typing import Callable, Union

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model: Callable,
        device: Union[str, torch.device],
        optimizer: Callable,
        criterion: Callable,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None

    def fit(
        self,
        epochs: int,
        train_loader: Callable,
        valid_loader: Callable,
        save_path: str,
        patience: int,
    ):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch,
                train_loss,
                train_time,
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch,
                valid_loss,
                valid_auc,
                valid_time,
            )

            if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                    "auc improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score,
                    valid_loss,
                    self.lastmodel,
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message(
                    "\nValid auc didn't improve last {} epochs.", patience
                )
                break

    def train_epoch(self, train_loader: Callable):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            wandb.log({"Training Loss": loss})

            self.optimizer.step()

            message = "Train Step {}/{}, train_loss: {:.4f}"
            self.info_message(
                message, step, len(train_loader), sum_loss / step, end="\r"
            )

        return sum_loss / len(train_loader), int(time.time() - t)

    def valid_epoch(self, valid_loader: Callable):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()

                wandb.log({"Validation Loss": loss})

                y_all.extend(batch["y"].tolist())
                outputs_all.extend(outputs.tolist())

            message = "Valid Step {}/{}, valid_loss: {:.4f}"
            self.info_message(
                message, step, len(valid_loader), sum_loss / step, end="\r"
            )

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)

        return sum_loss / len(valid_loader), auc, int(time.time() - t)

    def save_model(self, n_epoch: int, save_path: str, loss: float, auc: float):
        self.lastmodel = f"{save_path}-e{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message: str, *args, end="\n"):
        print(message.format(*args), end=end)
