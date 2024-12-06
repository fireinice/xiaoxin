import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from .base_model import BaseModelModule
from src.architectures import DrugProteinAttention
import logging
logging.basicConfig(filename='val.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class MorganAttention(BaseModelModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=52,
        loss_type="OR",
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
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label.to(torch.float32))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label.to(torch.float32))
        self.log("val/loss", loss)
        if self.loss_type=="OR":
            pred = self.ordinal_regression_predict(pred)
        else:
            pred = pred
        result = {"loss": loss, "preds": pred, "target": label}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        gathered_outputs = self.all_gather(self.validation_step_outputs)
        all_loss = torch.stack([x["loss"] for x in gathered_outputs]).mean()
        all_preds = torch.concat([x["preds"] for x in gathered_outputs])
        all_target = torch.concat([x["target"] for x in gathered_outputs])
        all_preds = all_preds.view(-1, all_preds.size(-1))
        all_target = all_target.view(-1)
        self.print(f"*****Epoch {self.current_epoch}*****")
        self.print(f"loss: {all_loss}")
        for name, metric in self.metrics.items():
            value = metric(all_preds, all_target)
            if np.isscalar(value):
                self.log(f"val/{name}", value)
            logging.info(f"val/{name}: {value}")
            if name == "ConfusionMatrix":
                confusion_matrix_np = value.cpu().numpy()
                np.savetxt("confusion_matrix.csv", confusion_matrix_np, delimiter=",", fmt="%d")
            self.print(f"val/{name}: {value}")
        self.validation_step_outputs.clear()

