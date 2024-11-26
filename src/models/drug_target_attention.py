import torch
import torchtune
from torch import nn, Tensor
from transformers import AutoModel

from src.architectures import ChemBertaProteinAttention, MLP
from src.models.base_model import BaseModelModule


# 三维的Chemberta和三维的Porbert  从本地加载三维的Chembert的特征
class ChemBertaProteinAttentionPreEncoded(nn.Module):
    def __init__(
        self,
        drug_shape=384,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        num_classes=2,
        loss_type="CE",
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.loss_type = loss_type
        self.head_size = 64
        self.num_heads = int(self.latent_dimension / self.head_size)

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
        )
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
        )
        self.input_norm = nn.LayerNorm(latent_dimension)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dimension, num_heads=self.num_heads, dropout=0.1, batch_first=True
        )
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.rpe = torchtune.modules.RotaryPositionalEmbeddings(self.head_size)

        self.mlp = MLP(latent_dimension * 2, 512, 256)

        if classify:
            if self.num_classes == 2:
                self.predict_layer = nn.Sequential(
                    nn.Linear(256, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                if self.loss_type == "OR":
                    self.predict_layer = nn.Sequential(
                        nn.Linear(256, self.num_classes - 1, bias=True), nn.Sigmoid()
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
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_att_mask(self, drug: Tensor):
        batch, seq, dim = drug.shape
        mask = torch.mean(drug, dim=-1)
        mask = torch.where(mask != 0.0, False, True)
        return mask

    def ordinal_regression_predict(self, predict):
        predict = (predict > 0.5).sum(dim=1)
        predict = nn.functional.one_hot(predict, num_classes=self.num_classes).to(
            torch.float32
        )
        return predict

    def align_embedding(
        self, embedding: Tensor, projector: nn.Sequential, dim: int
    ) -> Tensor:
        if dim != self.latent_dimension:
            projection = projector(embedding)
        else:
            projection = embedding
        return self.input_norm(projection)

    def cross_attetion(self, q: Tensor, k: Tensor, v: Tensor, q_mask, k_mask) -> Tensor:
        attention, _ = self.cross_attn(q, k, v, key_padding_mask=k_mask)
        attention = attention * (~q_mask).unsqueeze(-1).float()
        # max pool along sentence dimension
        output = self.max_pool(attention.permute(0, 2, 1)).squeeze()
        return output

    def position(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # https://raw.githubusercontent.com/mohitpg/LLMs-from-scratch/445cb0545ab53c7bf416ca3aa47bf44ed9f12566/LLAMA.py
        B, T, D = x.shape
        q = k = x.view((B, T, self.num_heads, self.head_size))
        q = self.rpe(q)
        k = self.rpe(k)
        q = q.view((B, T, D))
        k = k.view((B, T, D))
        return (q, k)

    def forward(self, drug: Tensor, target: Tensor, is_train=True):
        drug_projection = self.align_embedding(
            drug, self.drug_projector, self.drug_shape
        )
        target_projection = self.align_embedding(
            target, self.target_projector, self.target_shape
        )

        target_att_mask = self.get_att_mask(target)
        drug_att_mask = self.get_att_mask(drug)
        drug_query, drug_key = self.position(drug_projection)
        target_query, target_key = self.position(target_projection)
        drug_output = self.cross_attetion(
            drug_query,
            target_key,
            target_projection,
            drug_att_mask,
            target_att_mask,
        )
        target_output = self.cross_attetion(
            target_query,
            drug_key,
            drug_projection,
            target_att_mask,
            drug_att_mask,
        )

        out_embedding = torch.concat([drug_output, target_output], dim=-1)

        x = self.mlp(out_embedding)
        predict = self.predict_layer(x)

        predict = torch.squeeze(predict, dim=-1)

        if is_train == False and self.loss_type == "OR":
            return self.ordinal_regression_predict(predict)
        else:
            return predict


class ChemBertaProteinAttention(ChemBertaProteinAttentionPreEncoded):
    def __init__(self, *args, finetune=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune = finetune
        self.drug_model = AutoModel.from_pretrained("./models/chemberta")

    def forward(
        self,
        drug_input_ids: Tensor,
        drug_att_masks: Tensor,
        target: Tensor,
        is_train=True,
    ):
        drug = self.drug_model(
            input_ids=drug_input_ids, attention_mask=drug_att_masks
        ).last_hidden_state
        if not self.finetune:
            drug = drug.detach()
        super().forward(drug, target, is_train)


class ChemBertaProteinBert(ChemBertaProteinAttentionPreEncoded):
    def forward(
        self,
        drug_input_ids: Tensor,
        drug_att_masks: Tensor,
        target: Tensor,
        is_train=True,
    ):
        drug = self.drug_model(
            input_ids=drug_input_ids, attention_mask=drug_att_masks
        ).last_hidden_state
        if not self.finetune:
            drug = drug.detach()
        return super().forward(drug, target, is_train)


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
        fine_tune=False,
    ):
        super().__init__(
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr
        )
        if not fine_tune:
            self.model = ChemBertaProteinAttentionPreEncoded(
                drug_dim,
                target_dim,
                latent_dim,
                classify=classify,
                num_classes=num_classes,
                loss_type=loss_type,
            )
        else:
            self.model = ChemBertaProteinAttention(
                drug_dim,
                target_dim,
                latent_dim,
                classify=classify,
                num_classes=num_classes,
                loss_type=loss_type,
                finetune=True,
            )
        self.lr_t0 = lr_t0
        self.validation_step_outputs = []

    def forward(self, drug, target):
        if isinstance(drug, dict):
            return self.model(drug["drug_input_ids"], drug["drug_att_masks"], target)
        else:
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
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        preds = torch.concat([x["preds"] for x in self.validation_step_outputs])
        target = torch.concat([x["target"] for x in self.validation_step_outputs])
        self.print(f"*****Epoch {self.current_epoch}*****")
        self.print(f"loss:{avg_loss}")
        for name, metric in self.metrics.items():
            value = metric(preds, target)
            self.log(f"val/{name}", value)
            self.print(f"val/{name}: {value}")
        self.validation_step_outputs.clear()
