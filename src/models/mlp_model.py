import torch
from src.architectures import MLP
from src.models.morgan_chembert_model import MorganChemBertAttention
import torch.nn.functional as F



class MorganChemBertMlp(MorganChemBertAttention):
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