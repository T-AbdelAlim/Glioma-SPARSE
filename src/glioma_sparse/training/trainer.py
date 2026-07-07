import torch
import csv
import time
import numpy as np
from pathlib import Path

from glioma_sparse.evaluation.metrics import compute_classification_metrics


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        output_dir,
        early_stopping_metric="auc",
        patience=12,
        min_delta=1e-4,
        save_every_epoch=False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "log.csv"

        # Early stopping
        self.early_stopping_metric = early_stopping_metric
        self.patience = patience
        self.min_delta = min_delta

        # Saving
        self.save_every_epoch = save_every_epoch

        # --------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "acc": [],
            "f1": [],
            "auc": []
        }

        # --------------------------------------------------------
        # INIT CSV
        # --------------------------------------------------------
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "accuracy",
                "f1",
                "auc",
                "epoch_time_sec"
            ])

    # --------------------------------------------------------
    # TRAIN ONE EPOCH
    # --------------------------------------------------------
    def train_one_epoch(self):

        self.model.train()
        total_loss = 0.0

        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------
    def validate(self):

        self.model.eval()

        all_probs = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for imgs, labels in self.val_loader:

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                total_loss += loss.item()

        if len(all_probs) == 0:
            return total_loss, {
                "accuracy": 0.0,
                "f1": 0.0,
                "auc": 0.0
            }, None, None

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_classification_metrics(all_probs, all_labels)

        return (
            total_loss / len(self.val_loader),
            metrics,
            all_probs,
            all_labels
        )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    def train(self, num_epochs):

        best_auc = 0.0
        best_f1 = 0.0
        best_acc = 0.0

        best_metric = -float("inf")
        epochs_no_improve = 0

        start_time_total = time.time()

        final_probs = None
        final_labels = None

        for epoch in range(num_epochs):

            print("\nEpoch {}/{}".format(epoch + 1, num_epochs))

            start_time_epoch = time.time()

            train_loss = self.train_one_epoch()
            val_loss, metrics, probs, labels = self.validate()

            epoch_time = time.time() - start_time_epoch

            print("Train loss: {:.4f}".format(train_loss))
            print("Val loss:   {:.4f}".format(val_loss))
            print("Acc: {:.3f} | F1: {:.3f} | AUC: {:.3f}".format(
                metrics["accuracy"],
                metrics["f1"],
                metrics["auc"]
            ))
            print("Epoch time: {:.2f} sec".format(epoch_time))

            # ----------------------------------------------------
            # LOG CSV
            # ----------------------------------------------------
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    train_loss,
                    val_loss,
                    metrics["accuracy"],
                    metrics["f1"],
                    metrics["auc"],
                    epoch_time
                ])

            # ----------------------------------------------------
            # SAVE CHECKPOINTS
            # ----------------------------------------------------
            torch.save(self.model.state_dict(), self.output_dir / "last.pth")

            if self.save_every_epoch:
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "epoch_{}.pth".format(epoch + 1)
                )

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(self.model.state_dict(), self.output_dir / "best_auc.pth")
                print("Saved best AUC model")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(self.model.state_dict(), self.output_dir / "best_f1.pth")
                print("Saved best F1 model")

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save(self.model.state_dict(), self.output_dir / "best_acc.pth")
                print("Saved best Accuracy model")

            # ----------------------------------------------------
            # EARLY STOPPING
            # ----------------------------------------------------
            current_metric = metrics[self.early_stopping_metric]

            if current_metric > best_metric + self.min_delta:
                best_metric = current_metric
                epochs_no_improve = 0

                final_probs = probs
                final_labels = labels
            else:
                epochs_no_improve += 1

            print("Early stopping: {}/{}".format(
                epochs_no_improve,
                self.patience
            ))

            # ----------------------------------------------------
            # HISTORY
            # ----------------------------------------------------
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["acc"].append(metrics["accuracy"])
            self.history["f1"].append(metrics["f1"])
            self.history["auc"].append(metrics["auc"])

            if epochs_no_improve >= self.patience:
                print("\nEarly stopping triggered")
                break

        # --------------------------------------------------------
        # FINAL STATS
        # --------------------------------------------------------
        total_time = time.time() - start_time_total

        print("\n=== TRAINING COMPLETE ===")
        print("Total time: {:.2f} sec".format(total_time))
        print("Best AUC: {:.3f}".format(best_auc))
        print("Best F1: {:.3f}".format(best_f1))
        print("Best Acc: {:.3f}".format(best_acc))

        print("AUC per hour: {:.3f}".format(
            best_auc / (total_time / 3600.0)
        ))

        return self.history, final_probs, final_labels