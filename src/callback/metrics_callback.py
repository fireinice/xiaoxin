import pandas as pd
import torch
import torchmetrics
import numpy as np
import logging
from pytorch_lightning import Callback
from transformers import trainer


class MetricsCallback(Callback):
    def __init__(self, num_classes, classify):
        super().__init__()
        self.num_classes = num_classes
        self.classify = classify
        if self.num_classes < 3:
            if self.classify:
                self.metrics = {
                    "class_accuracy": torchmetrics.classification.BinaryAccuracy(),
                    "class_recall": torchmetrics.classification.BinaryRecall(),
                    "class_precision": torchmetrics.classification.BinaryPrecision(),
                    "F1Score": torchmetrics.classification.BinaryF1Score(),
                    "ConfusionMatrix": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                }
            else:
                self.metrics = {
                    "mse": torchmetrics.MeanSquaredError(),
                    "pcc": torchmetrics.PearsonCorrCoef(),
                }
        else:
            self.metrics = {
                "class_accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes,
                                                                                 average=None),
                "class_recall": torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes,
                                                                             average=None),
                "class_precision": torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes,
                                                                                   average=None),
                "F1Score": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average=None),
                "F1Score_Average": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes,
                                                                                 average='weighted'),
                "ConfusionMatrix": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
            }

        self.metrics = torch.nn.ModuleDict(self.metrics)

    def save_to_txt(self, drug, target, label, pred, file_path):
        drug = drug.cpu().numpy() if torch.is_tensor(drug) else drug
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        with open(file_path, 'w') as f:
            f.write("drug, target, label, pred\n")
            for d, t, l, p in zip(drug, target, label, pred):
                f.write(f"{d}, {t}, {l}, {p}\n")

    def on_validation_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        gathered_outputs = pl_module.all_gather(pl_module.validation_step_outputs)
        all_loss = torch.stack([x["loss"] for x in gathered_outputs]).mean()
        all_preds = torch.concat([x["preds"] for x in gathered_outputs])
        all_target = torch.concat([x["target"] for x in gathered_outputs])
        if self.classify:
            all_preds = all_preds.view(-1, all_preds.size(-1))
        else:
            all_preds = all_preds.view(-1)
        all_target = all_target.view(-1)
        pl_module.print(f"*****Epoch {pl_module.current_epoch}*****")
        pl_module.print(f"loss: {all_loss}")
        for name, metric in self.metrics.items():
            metric = metric.to(device)
            value = metric(all_preds, all_target)
            logging.info(f"val/{name}: {value}")
            if name == "F1Score_Average":
                pl_module.log(f"val/{name}", value)
            elif name == "ConfusionMatrix":
                confusion_matrix_np = value.cpu().numpy()
                np.savetxt(f"confusion_matrix.csv", confusion_matrix_np, delimiter=",", fmt="%d")
            pl_module.print(f"val/{name}: {value}")
        pl_module.validation_step_outputs.clear()
