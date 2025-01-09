import torch
from torch import nn
from src.models.morgan_chembert_model import MorganChemBertAttention
import torch.nn.functional as F


class MorganChemBertMhAttention(MorganChemBertAttention):
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
        ensemble_learn = False,
        lr_t0=10,
    ):
        super().__init__(
            drug_dim, drug_dim_two,target_dim, latent_dim, classify, num_classes, loss_type, lr , ensemble_learn,lr_t0
        )
        self.mh_attn_one = nn.MultiheadAttention(
            embed_dim=self.latent_dimension, num_heads=16, dropout=0.1, batch_first=True
        )
        self.mh_attn_two = nn.MultiheadAttention(
            embed_dim=self.latent_dimension, num_heads=16, dropout=0.1, batch_first=True
        )

    def get_att_mask(self, target: torch.Tensor):

        b, n, d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)

        return mask

    def forward(self,
                drug: torch.Tensor,
                target: torch.Tensor,):

        b, n, target_d = target.shape

        drug_projection_one = self.drug_projector(drug['drugs_one'])
        drug_projection_one = torch.tanh(drug_projection_one)
        drug_projection_two = self.drug_projector_two(drug['drugs_two'])
        drug_projection_two = torch.tanh(drug_projection_two)
        if target_d != self.latent_dimension:
            target_projection = self.target_projector(target)
            target_projection = torch.tanh(target_projection)
        else:
            target_projection = target

        drug_projection_one = drug_projection_one.unsqueeze(1)
        drug_projection_two = drug_projection_two.unsqueeze(1)

        att_mask = self.get_att_mask(target_projection)
        input_one = self.input_norm(drug_projection_one)
        input_two = self.input_norm_two(drug_projection_two)

        input_one, _ = self.mh_attn_one(
            input_one,
            target_projection,
            target_projection,
            key_padding_mask = att_mask
        )
        input_two, _ = self.mh_attn_two(
            input_two,
            target_projection,
            target_projection,
            key_padding_mask=att_mask
        )

        out_embedding_one = self.pooler(input_one)
        out_embedding_two = self.pooler_two(input_two)

        weights = F.softmax(torch.stack([self.weight_one, self.weight_two]), dim=0)
        weight_one, weight_two = weights[0], weights[1]

        out_embedding = weight_one * out_embedding_one + weight_two * out_embedding_two
        return self.classifier_forward(out_embedding)