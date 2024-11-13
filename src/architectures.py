import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pk
from types import SimpleNamespace
from typing import Optional

from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from .utils import get_logger
from .featurizers.protein import FOLDSEEK_MISSING_IDX

from torch.nn.utils.rnn import pad_sequence
from torch import nn,einsum
import math
from transformers import AutoTokenizer, AutoModel, pipeline


logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(
            torch.FloatTensor([float(k)]), requires_grad=False
        )
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


#######################
# Model Architectures #
#######################


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCoembedding_FoldSeek(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=1024,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):

        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(
            foldseek_indices
        ).mean(dim=1)

        full_target_embedding = torch.cat(
            [plm_embedding, foldseek_embedding], dim=1
        )
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembedding_FoldSeekX(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=512,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        # self.projector_dropout = nn.Dropout(p=0.2)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):

        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(
            foldseek_indices
        ).mean(dim=1)

        full_target_embedding = torch.cat(
            [plm_embedding, foldseek_embedding], dim=1
        )
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=100,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, 1, bias=True),
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_layers(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        return y


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2), activation()
        )
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()

class MLP(nn.Module):

    def __init__(self, inputdim: int, hiddendim:int, outdim:int) -> None:
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(inputdim, hiddendim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hiddendim, outdim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        return self.mlp(inputs)



def FeedForward(dim : int, mult : int= 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

    
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim : int,
        dim_head : int = 64,
        heads : int = 8
    ):
        super(PerceiverAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    

    def rearrange(self, x : torch.Tensor, num_head) -> torch.Tensor:

        x = x.view(x.shape[0], x.shape[1], num_head, -1).permute(0, 2, 1, 3)
        return x


    def forward(self, x : torch.Tensor, latents : torch.Tensor, x_att_mask: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """


       

        b, n, d = latents.shape

        latent_mask = torch.zeros(b, n).to(x_att_mask.device)
        latent_mask = torch.where(latent_mask == 0, False, True)
        mask = torch.concat([x_att_mask, latent_mask],dim=1)

        # 2. 扩展到 (batch_size, sequence_length, sequence_length)

        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, n, -1)

        mask = torch.where(mask==True, float("-inf"), 1)

        mask = mask[:, None,:,:]


        


        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        #, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = self.rearrange(q, h)
        k = self.rearrange(k, h)
        v = self.rearrange(v, h)


        q = q * self.scale

        # attention

        sim = einsum('b h i d, b h j d  -> b h i j', q, k)
        sim = sim - sim.max(dim = -1, keepdim = True)[0]
        sim += mask
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        #out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        b,h,n,d = out.shape

        out = out.permute(0, 2, 1, 3).reshape(b, n, h*d)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        ff_mult: int = 4
    ) -> None:
        super(PerceiverResampler, self).__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)
    

    def forward(self, x : torch.Tensor, att_mask: torch.Tensor)->torch.Tensor:

        b = x.shape[0]

        latents = self.latents.unsqueeze(0).repeat(b,1,1)

        for attn, ff in self.layers:
            latents = attn(x, latents,att_mask) + latents
            latents = ff(latents) + latents

        latents = self.norm(latents)
        return latents


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, max_position_embeddings, hidden_size, type_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
       
        input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, 0 : seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664

       
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


        
class DrugProteinAttention(nn.Module):

    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        num_classes=2,
        loss_type="CE"
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify  
        self.num_classes=num_classes
        self.loss_type=loss_type


        self.pooler = BertPooler(latent_dimension) 


        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension)
        )
        #nn.init.xavier_normal_(self.drug_projector[0].weight)
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension)
        )

        #self.position = PositionalEncoding(d_model=latent_dimension,max_len=latent_dimension)
        #nn.init.xavier_normal_(self.target_projector[0].weight)

        self.input_norm = nn.LayerNorm(latent_dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dimension, nhead=16,batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        

        self.mlp = MLP(latent_dimension, 512, 256)

        if classify:

          
            if self.num_classes == 2:
                 self.predict_layer = nn.Sequential(
                 nn.Linear(256, 1, bias=True),
                 nn.Sigmoid(),
            )
            else:

                if self.loss_type == 'OR':
                    self.predict_layer = nn.Sequential(
                    nn.Linear(256, self.num_classes-1, bias=True),
                    nn.Sigmoid()
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
        initializer_range = 0.02
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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
    

    def get_att_mask(self, target:torch.Tensor):

        b,n,d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask !=0.0, False, True)

        drug_mask = torch.zeros(b, 1).to(mask.device)
        drug_mask = torch.where(drug_mask == 0, False, True)
        mask = torch.concat([drug_mask, mask],dim=1)
        return mask

    def ordinal_regression_predict(self, predict):

        predict =  (predict > 0.5).sum(dim=1)

        predict = torch.nn.functional.one_hot(predict,num_classes=self.num_classes).to(torch.float32)
        return predict
    def forward(self, drug : torch.Tensor, target:torch.Tensor,is_train=True):

        b, d = drug.shape 

        b,n,d = target.shape 

        drug_projection = self.drug_projector(drug)
        #target_projection = self.target_projector(target)
        target_projection = target

        drug_projection = drug_projection.unsqueeze(1)
        inputs = torch.concat([drug_projection, target_projection], dim=1)


        inputs = self.input_norm(inputs)

        #inputs = self.position(inputs)

        att_mask = self.get_att_mask(target)

        outputs = self.transformer_encoder(inputs,src_key_padding_mask=att_mask)
        out_embedding = self.pooler(outputs)

        #out_embedding = torch.max(outputs, dim=1)[0]

        x = self.mlp(out_embedding)
        predict = self.predict_layer(x)

        predict = torch.squeeze(predict,dim=-1)
        if (is_train==False and self.loss_type=="OR"):
            return self.ordinal_regression_predict(predict) 
        else:
            return predict


class DrugProteinMLP(nn.Module):

    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify   


        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape,latent_dimension)
        )

        proj_dim = 256
        num_head = 16
        num_latents = 16

        self.proj = nn.Linear(latent_dimension,proj_dim)

        self.att_drop = nn.Dropout(0.1)

        self.pooler = PerceiverResampler(
            dim=target_shape,
            depth=3,
            dim_head=int(target_shape/num_head),
            heads=num_head,
            num_latents=num_latents)




        self.mlp = MLP(latent_dimension + proj_dim*num_latents, 512, 256)
        if classify:

            self.predict_layer = nn.Sequential(
                 nn.Linear(256, 1, bias=True),
                 nn.Sigmoid(),
            )
        else:
            self.predict_layer = nn.Sequential(
                 nn.Linear(256, 1, bias=True),
                 nn.ReLU(),
            ) 

        self.apply(self._init_weights)

    

    def _init_weights(self, module):
        """Initialize the weights"""
        initializer_range = 0.02
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.xavier_normal_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.xavier_normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    

    def scaled_dot_product_attention(self,query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        

        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        
        
       
        attn_weight = query @ key.transpose(-2, -1) * torch.tensor(scale_factor).to(query.dtype)
        #attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.att_drop(attn_weight)
        return attn_weight @ value

    
    def get_att_mask(self, target:torch.Tensor):

        b,n,d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask !=0.0, False, True)
        return mask

    def forward(self, drug : torch.Tensor, target:torch.Tensor):


        drug_projection = self.drug_projector(drug)

        attention_mask = self.get_att_mask(target)

        target = self.pooler(target,attention_mask)
        drug_projection = drug_projection.unsqueeze(1)

        drug_embedding = self.scaled_dot_product_attention(drug_projection, target,target) #b, d
        drug_embedding = drug_embedding.squeeze(dim=1)
       
        target_embedding = self.scaled_dot_product_attention(target, drug_projection,drug_projection)

        target_embedding = self.proj(target_embedding)

        target_embedding = target_embedding.flatten(start_dim=-2,end_dim=-1)
        
        #inputs = torch.concat([drug_embedding, target_embedding], dim=-1)

        x = self.mlp(drug_embedding)
        predict = self.predict_layer(x)

        predict = torch.squeeze(predict,dim=-1)
        return predict

class ChemBertaProteinAttention(nn.Module):

    def __init__(
            self,
            drug_shape=384,
            target_shape=1024,
            latent_dimension=384,
            latent_activation=nn.ReLU,
            latent_distance="Cosine",
            classify=True,
            num_classes=2,
            loss_type="CE"
    ):
        super().__init__()

        
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.loss_type = loss_type

        self.pooler = BertPooler(latent_dimension)

       
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension)
        )
        # nn.init.xavier_normal_(self.drug_projector[0].weight)
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension)
        )

        # nn.init.xavier_normal_(self.drug_projector[0].weight)
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension)
        )

        # self.position = PositionalEncoding(d_model=latent_dimension,max_len=latent_dimension)
        # nn.init.xavier_normal_(self.target_projector[0].weight)
        
       #self.target_model = AutoModel.from_pretrained('./models/probert')
        self.drug_model = AutoModel.from_pretrained('./models/chemberta')

        self.input_norm = nn.LayerNorm(latent_dimension)

        self.cross_attn = nn.MultiheadAttention(embed_dim=self.latent_dimension, num_heads=16,dropout=0.1,batch_first=True)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.mlp = MLP(latent_dimension*2, 512, 256)

        if classify:

            if self.num_classes == 2:
                 self.predict_layer = nn.Sequential(
                 nn.Linear(256, 1, bias=True),
                 nn.Sigmoid(),
                )
            else:

                if self.loss_type == 'OR':
                    self.predict_layer = nn.Sequential(
                    nn.Linear(256, self.num_classes-1, bias=True),
                    nn.Sigmoid()
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
        #for param in self.drug_model.parameters(): param.requires_grad = True
        #for param in self.target_model.parameters(): param.requires_grad = True


    def _init_weights(self, module):
        """Initialize the weights"""
        initializer_range = 0.02
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
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

    def get_att_mask(self, drug:torch.Tensor):
        batch, seq, dim = drug.shape

        mask = torch.mean(drug, dim=-1)
        mask = torch.where(mask != 0.0, False, True)
        return mask

    def ordinal_regression_predict(self, predict):

        predict =  (predict > 0.5).sum(dim=1)
        predict = torch.nn.functional.one_hot(predict,num_classes=self.num_classes).to(torch.float32)
        return predict

    def forward(self, 
                drug_input_ids: torch.Tensor, 
                drug_att_masks: torch.Tensor,
                target:torch.Tensor,
                is_train=True):

        drug_embedding = self.drug_model(input_ids=drug_input_ids,attention_mask=drug_att_masks).last_hidden_state
            #target = self.target_model(input_ids=target_input_ids,
                                                 #attention_mask=target_att_masks).last_hidden_state

        drug_embedding = drug_embedding.detach()
        drug_projection = self.drug_projector(drug_embedding)
        target_projection = self.target_projector(target)

        target_att_mask = self.get_att_mask(target)



        # drug_projection = drug_projection.unsqueeze(1)
        # inputs = torch.concat([drug_projection, target_projection], dim=1)

        drug_projection = self.input_norm(drug_projection)
        target_projection = self.input_norm(target_projection)

        # inputs = self.position(inputs)

        drug_att_masks = ~drug_att_masks.bool()
        

        drug_output , _ = self.cross_attn(drug_projection,target_projection,target_projection,key_padding_mask=target_att_mask)
        target_ouput, _ = self.cross_attn(target_projection,drug_projection,drug_projection,key_padding_mask=drug_att_masks)

        # out_embedding = self.pooler()

        drug_output = self.max_pool(drug_output.permute(0, 2, 1)).squeeze()
        target_ouput = self.max_pool(target_ouput.permute(0, 2, 1)).squeeze()

        out_embedding = torch.concat([drug_output,target_ouput],dim=-1)

        x = self.mlp(out_embedding)
        predict = self.predict_layer(x)

        predict = torch.squeeze(predict, dim=-1)

        if (is_train==False and self.loss_type=="OR"):
            return self.ordinal_regression_predict(predict) 
        else:
            return predict
