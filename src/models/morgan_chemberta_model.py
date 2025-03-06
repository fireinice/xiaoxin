import torch
from torch import nn
from src.architectures import BertPooler, MLP
from src.models.morgan_attention_model import MorganAttention
import torch.nn.functional as F



class MorganChemBertaAttention(MorganAttention):
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
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr , ensemble_learn,lr_t0
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

class MorganChemBertaMhAttention(MorganChemBertaAttention):
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

class MorganChemBertaMlp(MorganChemBertaAttention):
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
        self.mlp_one = MLP(self.latent_dimension, self.latent_dimension, self.latent_dimension)
        self.mlp_two = MLP(self.latent_dimension, self.latent_dimension, self.latent_dimension)

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
        input_one = torch.concat([drug_projection_one, target_projection], dim=1)
        input_two = torch.concat([drug_projection_two, target_projection], dim=1)

        input_one = self.input_norm(input_one)
        input_two = self.input_norm_two(input_two)

        output_one = self.mlp_one(input_one)
        output_two = self.mlp_two(input_two)

        out_embedding_one = self.pooler(output_one)
        out_embedding_two = self.pooler_two(output_two)

        weights = F.softmax(torch.stack([self.weight_one, self.weight_two]), dim=0)
        weight_one, weight_two = weights[0], weights[1]

        out_embedding = weight_one * out_embedding_one + weight_two * out_embedding_two
        return self.classifier_forward(out_embedding)

class MorganChemBertaAttentionFull(MorganChemBertaAttention):
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

    def get_att_mask(self, target: torch.Tensor):
        b, n, d = target.shape
        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)
        drug_mask = torch.zeros(b, 2).to(mask.device)
        drug_mask = torch.where(drug_mask == 0, False, True)
        mask = torch.concat([drug_mask, mask], dim=1)
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
        input_one = torch.concat([drug_projection_one, target_projection], dim=1)
        input = torch.concat([drug_projection_two,input_one], dim=1)

        input = self.input_norm(input)

        att_mask = self.get_att_mask(input)

        output = self.transformer_encoder(input, src_key_padding_mask=att_mask)

        out_embedding = output.mean(dim=1)

        return self.classifier_forward(out_embedding)