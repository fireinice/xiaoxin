import numpy as np
import torch
from .base_model import BaseModelModule
from src.architectures import DrugProteinAttention

class MorganAttention(BaseModelModule):
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
        self.model = DrugProteinAttention(
            drug_dim,
            target_dim,
            latent_dim,
            classify=classify,
            num_classes=num_classes,
            loss_type=loss_type,
        )
        self.lr_t0 = lr_t0
        self.validation_step_outputs = []

    def forward(self, drug, target,is_train=True):
        return self.model(drug, target,is_train=is_train)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.lr_t0
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def training_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target,True).to(torch.float64)
        loss = self.loss_fct(pred, label)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target,False).to(torch.float64)
        loss = self.loss_fct(pred, label)
        self.log("val/loss", loss)
        result = {"loss": loss, "preds": pred, "target": label}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        preds = torch.concat([x["preds"] for x in self.validation_step_outputs])
        target = torch.concat([x["target"] for x in self.validation_step_outputs])
        self.print(f"*****Epoch {self.current_epoch}*****")
        self.print(f"loss:{avg_loss}")
        for name, metric in self.metrics.items():
            value = metric(preds, target)
            if np.isscalar(value):
                self.log(f"val/{name}", value)
            self.print(f"val/{name}: {value}")
        self.validation_step_outputs.clear()
