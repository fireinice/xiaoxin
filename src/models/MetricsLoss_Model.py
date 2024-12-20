import pandas as pd
import torch
import torchmetrics
import numpy as np
import logging
from pytorch_lightning import Callback
from transformers import trainer


class MetricsLossCallback(Callback):
    def __init__(self, num_classes, classify, loss_type="CE"):
        super().__init__()
        self.num_classes = num_classes
        self.classify = classify
        self.loss_type = loss_type
        self.init_metrics_and_loss()

    def init_metrics_and_loss(self):
        if self.num_classes < 3:
            if self.classify:
                self.loss_fct = torch.nn.BCELoss()
                self.metrics = {
                    "class_accuracy": torchmetrics.classification.BinaryAccuracy(),
                    "class_recall": torchmetrics.classification.BinaryRecall(),
                    "class_precision": torchmetrics.classification.BinaryPrecision(),
                    "F1Score": torchmetrics.classification.BinaryF1Score(),
                    "ConfusionMatrix": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                }
            else:
                self.loss_fct = torch.nn.MSELoss()
                self.metrics = {
                    "mse": torchmetrics.MeanSquaredError(),
                    "pcc": torchmetrics.PearsonCorrCoef(),
                }
        else:
            if self.loss_type == "OR":
                self.loss_fct = self.ordinal_regression_loss

            elif self.loss_type == "CLM":
                self.loss_fct = self.clm_loss
            else:
                self.loss_fct = torch.nn.CrossEntropyLoss()
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

    def ordinal_regression_loss(self, y_pred, y_target):
        num_thresholds = y_pred.size(1)
        y_true_expanded = y_target.unsqueeze(1).repeat(1, num_thresholds)
        mask = (torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) < y_true_expanded).float()
        loss = torch.nn.BCELoss()(y_pred, mask)
        return loss

    def clm_loss(self, y_pred, y_true):

        eps = 1e-15
        y_true = y_true.unsqueeze(dim=-1)
        likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
        loss = -torch.log(likelihoods).mean()

        return loss

    def ordinal_regression_predict(self, predict):
        predict = (predict > 0.5).sum(dim=1)
        predict = torch.nn.functional.one_hot(predict, num_classes=self.num_classes).to(torch.float32)
        return predict

    def clm_predict(self, y_pred):
        num_thresholds = y_pred.size(1)
        mask = torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0)
        y_pred = torch.sum(y_pred * mask, dim=-1)
        y_pred = torch.round(y_pred).to(torch.int64)
        predict = torch.nn.functional.one_hot(y_pred, num_classes=num_thresholds).to(torch.float32)
        print(predict.shape)
        return predict

    def save_to_txt(self, drug, target, label, pred, file_path):
        drug = drug.cpu().numpy() if torch.is_tensor(drug) else drug
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        with open(file_path, 'w') as f:
            f.write("drug, target, label, pred\n")
            for d, t, l, p in zip(drug, target, label, pred):
                f.write(f"{d}, {t}, {l}, {p}\n")

    def on_train_start(self, trainer, pl_module):
        pl_module.loss_fct = self.loss_fct

    def on_validation_epoch_end(self, trainer, pl_module):
        device =  pl_module.device
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
            if np.isscalar(value):
                pl_module.log(f"val/{name}", value)
            logging.info(f"val/{name}: {value}")
            if name == "ConfusionMatrix":
                confusion_matrix_np = value.cpu().numpy()
                np.savetxt(f"confusion_matrix.csv", confusion_matrix_np, delimiter=",", fmt="%d")
            pl_module.print(f"val/{name}: {value}")
        pl_module.validation_step_outputs.clear()

    def on_predict_epoch_end(self, trainer, pl_module):
        gathered_outputs = pl_module.all_gather(pl_module.predict_step_outputs)
        all_preds = torch.concat([x["pred"] for x in gathered_outputs])
        all_proteome_ids = torch.concat([x["ID"] for x in gathered_outputs])
        all_preds = all_preds.view(-1, all_preds.size(-1)).cpu().numpy()
        all_proteome_ids = all_proteome_ids.view(-1).cpu().numpy()
        df = pd.DataFrame({
            'Prediction': all_preds.argmax(axis=1),
            'ID': all_proteome_ids.astype(int),
        })
        df.to_csv('predictions.csv', index=False)
        pl_module.print("Predictions saved to 'predictions.csv'.")