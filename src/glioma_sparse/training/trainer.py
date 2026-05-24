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
        output_dir
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

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = compute_classification_metrics(all_probs, all_labels)

        return total_loss / len(self.val_loader), metrics

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    def train(self, num_epochs):

        best_auc = 0.0
        best_f1 = 0.0
        best_acc = 0.0

        start_time_total = time.time()

        for epoch in range(num_epochs):

            print("\nEpoch {}/{}".format(epoch + 1, num_epochs))

            start_time_epoch = time.time()

            train_loss = self.train_one_epoch()
            val_loss, metrics = self.validate()

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
            # SAVE LAST
            # ----------------------------------------------------
            torch.save(
                self.model.state_dict(),
                self.output_dir / "last.pth"
            )

            # ----------------------------------------------------
            # SAVE BEST (AUC)
            # ----------------------------------------------------
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]

                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_auc.pth"
                )
                print("Saved best AUC model")

            # ----------------------------------------------------
            # SAVE BEST (F1)
            # ----------------------------------------------------
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]

                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_f1.pth"
                )
                print("Saved best F1 model")

            # ----------------------------------------------------
            # SAVE BEST (ACCURACY)
            # ----------------------------------------------------
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]

                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_acc.pth"
                )
                print("Saved best Accuracy model")

        total_time = time.time() - start_time_total

        print("\n=== TRAINING COMPLETE ===")
        print("Total time: {:.2f} sec".format(total_time))

        print("Best AUC: {:.3f}".format(best_auc))
        print("Best F1: {:.3f}".format(best_f1))
        print("Best Acc: {:.3f}".format(best_acc))

        print("AUC per hour: {:.3f}".format(
            best_auc / (total_time / 3600.0)
        ))