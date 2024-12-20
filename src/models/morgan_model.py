import numpy as np
import pandas as pd
import torch
from spacecutter.models import LogisticCumulativeLink
from torch import nn
from .base_model import BaseModelModule
from src.architectures import BertPooler, MLP
import logging
import time
log_filename = f"log_Morgan.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MorganAttention(BaseModelModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=2,
        loss_type="CLM",
        lr=1e-4,
        Ensemble_Learn=False,
        lr_t0=10,
    ):
        super().__init__(
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr,Ensemble_Learn
        )
        self.lr_t0 = lr_t0
        self.validation_step_outputs = []
        self.predict_step_outputs = []
        self.pooler = BertPooler(self.latent_dimension)

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, self.latent_dimension)
        )
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, self.latent_dimension)
        )
        self.input_norm = nn.LayerNorm(self.latent_dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dimension, nhead=16, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.mlp = MLP(self.latent_dimension, 512, 256)
        if self.classify:
            self.link = LogisticCumulativeLink(self.num_classes)

        if classify:
            if self.num_classes == 2:
                self.predict_layer = nn.Sequential(
                    nn.Linear(256, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                if self.loss_type == 'OR' :
                    if self.Ensemble_Learn:
                        self.predict_layer = nn.ModuleList([
                            nn.Sequential(
                                nn.Linear(256, 1, bias=True),
                                nn.Dropout(0.1),
                                nn.Sigmoid(),
                            )
                            for _ in range(self.num_classes - 1)
                        ])
                    else:
                        self.predict_layer = nn.Sequential(
                        nn.Linear(256, self.num_classes-1, bias=True),
                        nn.Sigmoid(),
                        )

                elif self.loss_type == "CLM":
                    self.predict_layer = nn.Sequential(
                        nn.Linear(256, 1, bias=True),
                    )
                else:
                    self.predict_layer = nn.Sequential(
                        nn.Linear(256, num_classes, bias=True),
                    )
        else:
            self.predict_layer = nn.Sequential(
                nn.Linear(256, 1, bias=True),
                nn.ReLU(),
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_att_mask(self, target: torch.Tensor):

        b, n, d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)

        drug_mask = torch.zeros(b, 1).to(mask.device)
        drug_mask = torch.where(drug_mask == 0, False, True)
        mask = torch.concat([drug_mask, mask], dim=1)
        return mask

    def classifier_forward(self, out_embedding ):
        x = self.mlp(out_embedding)
        if self.classify:
            if self.loss_type == 'OR' and self.Ensemble_Learn:
                predict = [classifier(x) for classifier in self.predict_layer]
                predict = torch.cat(predict, dim=1)
            elif self.loss_type == 'CLM':
                predict = self.predict_layer(x)
                predict = self.link(predict)
            else:
                predict = self.predict_layer(x)
        else:
            predict = self.predict_layer(x)
        predict = torch.squeeze(predict, dim=-1)
        return predict

    def forward(self, drug: torch.Tensor, target: torch.Tensor):

        b, d = drug.shape

        b, n, d = target.shape

        drug_projection = self.drug_projector(drug)
        drug_projection = torch.tanh(drug_projection)
        target_projection = target

        drug_projection = drug_projection.unsqueeze(1)
        inputs = torch.concat([drug_projection, target_projection], dim=1)

        inputs = self.input_norm(inputs)

        att_mask = self.get_att_mask(target)

        outputs = self.transformer_encoder(inputs, src_key_padding_mask=att_mask)
        out_embedding = self.pooler(outputs)
        return self.classifier_forward(out_embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
        if self.classify:
            if self.loss_type == "OR":
                pred = self.ordinal_regression_predict(pred)
            elif self.loss_type == 'CLM':
                pred = self.clm_predict(pred)
            else:
                pred = pred
        else:
            pred = pred
        result = {"loss": loss, "preds": pred, "target": label}
        self.validation_step_outputs.append(result)
        return result



