import pandas as pd
import torch
from torch import nn
from src.architectures import BertPooler
from src.models.morgan_model import MorganAttention
import torch.nn.functional as F
import logging
import time
log_filename = f"log_MorganChembert.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MorganChembertAttention(MorganAttention):
    def __init__(
        self,
        drug_dim=2048,
        drug_dim_two=384,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=2,
        loss_type="CE",
        lr=1e-4,
        Ensemble_Learn = False,
        lr_t0=10,
    ):
        super().__init__(
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr , Ensemble_Learn,lr_t0
        )
        self.drug_shape_two = drug_dim_two
        self.pooler_two = BertPooler(self.latent_dimension)
        self.drug_projector_two = nn.Sequential(
            nn.Linear(self.drug_shape_two, self.latent_dimension)
        )
        self.input_norm_two = nn.LayerNorm(self.latent_dimension)
        encoder_layer_two = nn.TransformerEncoderLayer(d_model=self.latent_dimension, nhead=16, batch_first=True)
        self.transformer_encoder_two = nn.TransformerEncoder(encoder_layer_two, num_layers=1)

        self.weight_one = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.weight_two = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self,
                drug: torch.Tensor,
                target: torch.Tensor,):

        drug_projection_one = self.drug_projector(drug['drugs_one'])
        drug_projection_two = self.drug_projector_two(drug['drugs_two'])
        target_projection = target

        drug_projection_one = drug_projection_one.unsqueeze(1)
        drug_projection_two = drug_projection_two.unsqueeze(1)
        input_one = torch.concat([drug_projection_one, target_projection], dim=1)
        input_two = torch.concat([drug_projection_two, target_projection], dim=1)

        input_one = self.input_norm(input_one)
        input_two = self.input_norm_two(input_two)

        att_mask = self.get_att_mask(target)

        output_one = self.transformer_encoder(input_one, src_key_padding_mask=att_mask)
        output_two = self.transformer_encoder_two(input_two, src_key_padding_mask=att_mask)

        out_embedding_one = self.pooler(output_one)
        out_embedding_two = self.pooler_two(output_two)

        weights = F.softmax(torch.stack([self.weight_one, self.weight_two]), dim=0)
        weight_one, weight_two = weights[0], weights[1]

        out_embedding = weight_one * out_embedding_one + weight_two * out_embedding_two
        return self.classifier_forward(out_embedding)

    def predict_step(self, batch, batch_idx):
        drug, target ,index = batch
        pred = self.forward(drug,target)
        if self.loss_type=="OR":
            pred = self.ordinal_regression_predict(pred)
        elif self.loss_type=='CLM':
            pred = self.clm_predict(pred)
        else:
            pred = pred
        result = {'ID':index,'pred': pred}
        self.predict_step_outputs.append(result)
        return result
