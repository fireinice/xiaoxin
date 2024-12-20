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

    def forward(self, drug, target):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, train_batch, batch_idx):
        raise NotImplementedError()