import torch
from torch import nn
from .base_model import BaseModelModule
from typing import Optional


class MLP(nn.Module):

    def __init__(self, inputdim: int, hiddendim: int, outdim: int) -> None:
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

        position_ids = self.position_ids[:, 0: seq_length]

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


# 二维的Chembert或者Morgan作为cls拼接三维的Protbert
class DrugProteinAttention(nn.Module):
    def __init__(
            self,
            drug_shape=2048,
            target_shape=1024,
            latent_dimension=1024,
            classify=True,
            num_classes=2,
            loss_type="CE",
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.num_classes = num_classes
        self.loss_type = loss_type

        self.pooler = BertPooler(latent_dimension)

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension)
        )
        # nn.init.xavier_normal_(self.drug_projector[0].weight)
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension)
        )

        # self.position = PositionalEncoding(d_model=latent_dimension,max_len=latent_dimension)
        # nn.init.xavier_normal_(self.target_projector[0].weight)

        self.input_norm = nn.LayerNorm(latent_dimension)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dimension, nhead=16, batch_first=True)

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
                        nn.Linear(256, self.num_classes - 1, bias=True),
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

    def get_att_mask(self, target: torch.Tensor):

        b, n, d = target.shape

        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)

        drug_mask = torch.zeros(b, 1).to(mask.device)
        drug_mask = torch.where(drug_mask == 0, False, True)
        mask = torch.concat([drug_mask, mask], dim=1)
        return mask

    def ordinal_regression_predict(self, predict):

        predict = (predict > 0.5).sum(dim=1)

        predict = torch.nn.functional.one_hot(predict, num_classes=self.num_classes).to(torch.float32)
        return predict

    def forward(self, drug: torch.Tensor, target: torch.Tensor, is_train=True):

        b, d = drug.shape

        b, n, d = target.shape

        drug_projection = self.drug_projector(drug)
        drug_projection = torch.tanh(drug_projection)
        # target_projection = self.target_projector(target)
        target_projection = target

        drug_projection = drug_projection.unsqueeze(1)
        inputs = torch.concat([drug_projection, target_projection], dim=1)

        inputs = self.input_norm(inputs)

        # inputs = self.position(inputs)

        att_mask = self.get_att_mask(target)

        outputs = self.transformer_encoder(inputs, src_key_padding_mask=att_mask)
        out_embedding = self.pooler(outputs)

        # out_embedding = torch.max(outputs, dim=1)[0]

        x = self.mlp(out_embedding)
        predict = self.predict_layer(x)

        predict = torch.squeeze(predict, dim=-1)

        if (is_train == False and self.loss_type == "OR"):
            return self.ordinal_regression_predict(predict)
        else:
            return predict


class MorganAttention(BaseModelModule):
    def __init__(
        self,
        drug_dim=2048,
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
        pred = self.forward(drug, target,True)
        loss = self.loss_fct(pred, label)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, target, label = train_batch  # target is (D + N_pool)
        pred = self.forward(drug, target,False)
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