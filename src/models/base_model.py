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
        Ensemble_Learn = False,
    ):
        super().__init__()
        self.drug_shape = drug_dim
        self.target_shape = target_dim
        self.latent_dimension = latent_dim
        self.classify = classify
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.lr = lr
        self.Ensemble_Learn = Ensemble_Learn
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
                self.loss_fct = self.ordinal_regression_loss

            elif self.loss_type=="CLM":
                self.loss_fct = self.clm_loss
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
        predict = torch.nn.functional.one_hot(predict,num_classes=self.num_classes).to(torch.float32)
        return predict

    def clm_predict(self, y_pred):
        num_thresholds = y_pred.size(1)
        mask = torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0)
        y_pred = torch.sum(y_pred * mask, dim=-1)
        y_pred = torch.round(y_pred).to(torch.int64)
        predict = torch.nn.functional.one_hot(y_pred,num_classes=num_thresholds).to(torch.float32)
        print(predict.shape)
        return predict

    def save_to_txt(self,drug, target, label, pred, file_path):
        drug = drug.cpu().numpy() if torch.is_tensor(drug) else drug
        target = target.cpu().numpy() if torch.is_tensor(target) else target
        label = label.cpu().numpy() if torch.is_tensor(label) else label
        pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        with open(file_path, 'w') as f:
            f.write("drug, target, label, pred\n")
            for d, t, l, p in zip(drug, target, label, pred):
                f.write(f"{d}, {t}, {l}, {p}\n")

    def forward(self, drug, target):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, train_batch, batch_idx):
        raise NotImplementedError()