import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import torchmetrics
from ..architectures import ChemBertaProteinAttention_Local
from .base_model import BaseModelModule


class DrugTargetAttention(BaseModelModule):
    def __init__(
        self,
        drug_dim=384,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=2,
        loss_type="CE",
        lr=1e-4,
        lr_t0=10,
    ):
        super().__init__(
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr
        )
        self.model = ChemBertaProteinAttention_Local(
            drug_dim,
            target_dim,
            latent_dim,
            classify=classify,
            num_classes=num_classes,
            loss_type=loss_type,
        )
        self.lr_t0 = lr_t0
        self.validation_step_outputs = []        

    def forward(self, drug, target):
        return self.model(drug, target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.lr_t0
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def training_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label)
        self.log("val/loss", loss)
        result = {"loss": loss, "preds": pred, "target": label}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        preds = torch.concat([x['preds'] for x in self.validation_step_outputs])
        target = torch.concat([x['target'] for x in self.validation_step_outputs]) 
        self.print(f"*****Epoch {self.current_epoch}*****")  
        self.print(f"loss:{avg_loss}")             
        for name, metric in self.metrics.items():
            value = metric(preds, target)
            self.log(f"val/{name}", value)
            self.print(f"val/{name}: {value}")
        self.validation_step_outputs.clear()            

    def test_step_end(self, outputs):
        for name, metric in self.metrics.items():
            metric(outputs["preds"], outputs["target"])
            self.log(f"test/{name}", metric)
