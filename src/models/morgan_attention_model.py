import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from src.architectures import BertPooler, MLP
import logging
log_filename = f"log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MorganAttention(pl.LightningModule):
    def __init__(
        self,
        drug_dim=2048,
        target_dim=1024,
        latent_dim=1024,
        classify=True,
        num_classes=2,
        loss_type="CLM",
        lr=1e-4,
        ensemble_learn=False,
        lr_t0=10,
    ):
        super().__init__()
        # Model Hyperparameters
        self.drug_shape = drug_dim
        self.target_shape = target_dim
        self.latent_dimension = latent_dim
        self.classify = classify
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.lr = lr
        self.ensemble_learn = ensemble_learn
        self.lr_t0 = lr_t0

        self.validation_step_outputs = []
        self.predict_step_outputs = []

        # Loss Function Setup
        if self.num_classes < 3:
            if self.classify:
                self.loss_fct = torch.nn.BCELoss()
            else:
                self.loss_fct = torch.nn.MSELoss()
        else:
            if self.loss_type == "OR":
                self.loss_fct = self.ordinal_regression_loss
            else:
                self.loss_fct = torch.nn.CrossEntropyLoss()

        # Define Model Components
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

        if classify:
            if self.num_classes == 2:
                self.predict_layer = nn.Sequential(
                    nn.Linear(256, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                if self.loss_type == 'OR':
                    if self.ensemble_learn:
                        self.predict_layer = nn.ModuleList([nn.Sequential(
                            nn.Linear(256, 1, bias=True),
                            nn.Dropout(0.1),
                            nn.Sigmoid(),
                        ) for _ in range(self.num_classes - 1)])
                    else:
                        self.predict_layer = nn.Sequential(
                            nn.Linear(256, self.num_classes-1, bias=True),
                            nn.Sigmoid(),
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

    def ordinal_regression_loss(self, y_pred, y_target):
        num_thresholds = y_pred.size(1)
        y_true_expanded = y_target.unsqueeze(1).repeat(1, num_thresholds)
        mask = (torch.arange(num_thresholds).to(y_pred.device).unsqueeze(0) < y_true_expanded).float()
        loss = torch.nn.BCELoss()(y_pred, mask)
        return loss


    def ordinal_regression_predict(self, predict):
        predict = (predict > 0.5).sum(dim=1)
        predict = torch.nn.functional.one_hot(predict, num_classes=self.num_classes).to(torch.float32)
        return predict

    def get_att_mask(self, target: torch.Tensor):
        b, n, d = target.shape
        mask = torch.mean(target, dim=-1)
        mask = torch.where(mask != 0.0, False, True)
        drug_mask = torch.zeros(b, 1).to(mask.device)
        drug_mask = torch.where(drug_mask == 0, False, True)
        mask = torch.concat([drug_mask, mask], dim=1)
        return mask

    def classifier_forward(self, out_embedding):
        x = self.mlp(out_embedding)
        if self.classify:
            if self.loss_type == 'OR' and self.ensemble_learn:
                predict = [classifier(x) for classifier in self.predict_layer]
                predict = torch.cat(predict, dim=1)
            else:
                predict = self.predict_layer(x)
        else:
            predict = self.predict_layer(x)
        predict = torch.squeeze(predict, dim=-1)
        return predict

    def forward(self, drug: torch.Tensor, target: torch.Tensor):
        b, d = drug.shape
        b, n, target_d = target.shape
        drug_projection = self.drug_projector(drug)
        drug_projection = torch.tanh(drug_projection)
        if target_d != self.latent_dimension:
            target_projection = self.target_projector(target)
            target_projection = torch.tanh(target_projection)
        else:
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
        drug, target, label = train_batch
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, train_batch, batch_idx):
        drug, target, label = train_batch
        pred = self.forward(drug, target)
        loss = self.loss_fct(pred, label)
        self.log("val/loss", loss)
        if self.classify and self.loss_type == "OR":
            pred = self.ordinal_regression_predict(pred)
        else:
            pred = pred
        result = {"loss": loss, "preds": pred, "target": label}
        self.validation_step_outputs.append(result)
        return result

    def predict_step(self, batch, batch_idx):
        drug,target,label= batch
        pred = self.forward(drug,target)
        if self.loss_type=="OR":
            pred = self.ordinal_regression_predict(pred)
        else:
            pred = pred
        result = { "preds": pred, "target": label}
        self.predict_step_outputs.append(result)
        return result

    def on_predict_epoch_end(self,):
        gathered_outputs = self.all_gather(self.predict_step_outputs)
        all_preds = torch.concat([x["pred"] for x in gathered_outputs])
        all_proteome_ids = torch.concat([x["label"] for x in gathered_outputs])
        all_preds = all_preds.view(-1, all_preds.size(-1)).cpu().numpy()
        all_proteome_ids = all_proteome_ids.view(-1).cpu().numpy()
        df = pd.DataFrame({
            'Prediction': all_preds.argmax(axis=1),
            'label': all_proteome_ids.astype(int),
        })
        df.to_csv(f'predictions.csv', index=False)
        self.print("Predictions saved to 'predictions.csv'.")
