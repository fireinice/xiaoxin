import torch
from torch import nn

import pytorch_lightning as pl
import torchmetrics


class BaseModelModule(pl.LightningModule):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=2,
        loss_type="CE",
        lr=1e-4,
    ):
        super().__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.classify = classify
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.lr = lr
        self.init_metrics_and_loss()

    def init_metrics_and_loss(self):
        if self.num_classes < 3:
            if self.classify:
                self.loss_fct = torch.nn.BCELoss()
            else:
                self.loss_fct = torch.nn.MSELoss()
            self.metrics = {
                "mse": torchmetrics.MeanSquaredError(),
                "pcc": torchmetrics.PearsonCorrCoef(),
            }
        else:
            if self.loss_type == "OR":
                self.loss_fct = self.map_ordinal_regression_loss
            else:
                self.loss_fct = torch.nn.CrossEntropyLoss()
            self.metrics = {
                "class_accuracy": torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes,average=None),
                "class_recall": torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes, average=None),
                "class_precision":torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes,average=None),
                "F1Score": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes,average=None),
                "F1Score_Average": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average='weighted'),
                "ConfusionMatrix":torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes),
                }

        # https://github.com/Lightning-AI/torchmetrics/issues/531
        self.metrics = torch.nn.ModuleDict(self.metrics)

    def ordinal_regression_loss(self, y_pred, y_target,factor=1):
        num_thresholds = y_pred.size(1)
        y_true_expanded = y_target.unsqueeze(1).repeat(1, num_thresholds)
        mask = (torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) < y_true_expanded).float()
        loss = torch.nn.BCELoss()(y_pred, mask)
        return loss

    def diff_ordinal_regression_loss(self, y_pred, y_target,factor=1):
        num_thresholds = y_pred.size(1)
        y_true_expanded = y_target.unsqueeze(1).repeat(1, num_thresholds)
        mask = (torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) < y_true_expanded).float()
        loss = torch.nn.BCELoss(reduction='none')(y_pred, mask)
        rank_diff = torch.abs(torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) - y_true_expanded)
        weight = (1 - mask) * (factor * rank_diff.float()) + mask
        weighted_loss = (loss * weight).mean()
        return weighted_loss

    def map_ordinal_regression_loss(self, y_pred, y_target):
        num_thresholds = y_pred.size(1)
        y_true_expanded = (y_target*10).unsqueeze(1).repeat(1, num_thresholds)
        mask = (torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) < y_true_expanded).float()
        loss = torch.nn.BCELoss()(y_pred, mask)
        return loss

    def forward(self, drug, target):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        raise NotImplementedError()
