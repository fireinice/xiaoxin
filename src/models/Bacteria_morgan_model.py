import torch
from torch import nn
from src.architectures import BertPooler
from src.models.morgan_model import MorganAttention
import torch.nn.functional as F
import logging
import time
log_filename = f"log_MorganChembert.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BacteriaMorganAttention(MorganAttention):
    def __init__(
        self,
        drug_dim=2048,
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
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dimension, num_heads=16, dropout=0.1, batch_first=True
        )

    def get_att_mask(self, target: torch.Tensor):

        b, n, d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)

        return mask

    def forward(self, drug: torch.Tensor, target: torch.Tensor):
        b, d = drug.shape
        b, n, d = target.shape
        drug_projection = self.drug_projector(drug)
        drug_projection = torch.tanh(drug_projection)
        target_projection = target
        drug_projection = drug_projection.unsqueeze(1)
        att_mask = self.get_att_mask(target_projection)
        drug_output, _ = self.cross_attn(
            drug_projection,
            target_projection,
            target_projection,
            key_padding_mask = att_mask
        )
        drug_output = drug_output.squeeze(dim=1)
        return self.classifier_forward(drug_output)



