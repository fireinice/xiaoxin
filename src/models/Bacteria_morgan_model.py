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
        ensemble_Learn = False,
        lr_t0=10,
    ):
        super().__init__(
            drug_dim, target_dim, latent_dim, classify, num_classes, loss_type, lr , ensemble_Learn,lr_t0
        )
        self.mh_attn = nn.MultiheadAttention(
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
        drug_output, _ = self.mh_attn(
            drug_projection,
            target_projection,
            target_projection,
            key_padding_mask = att_mask
        )
        drug_output = drug_output.squeeze(dim=1)
        return self.classifier_forward(drug_output)

    def predict_step(self, batch, batch_idx):
        drug, target , ids= batch
        pred = self.forward(drug,target)
        if self.loss_type=="OR":
            pred = self.ordinal_regression_predict(pred)
        elif self.loss_type=='CLM':
            pred = self.clm_predict(pred)
        else:
            pred = pred
        result = {'ID': ids, 'pred': pred}
        self.predict_step_outputs.append(result)
        return result

    def on_predict_epoch_end(self,result):
        gathered_outputs = self.all_gather(self.predict_step_outputs)
        all_preds = torch.concat([x["pred"] for x in gathered_outputs])
        all_proteome_ids = torch.concat([x["ID"] for x in gathered_outputs])
        all_preds = all_preds.view(-1, all_preds.size(-1)).cpu().numpy()
        all_proteome_ids = all_proteome_ids.view(-1).cpu().numpy()
        df = pd.DataFrame({
            'Prediction': all_preds.argmax(axis=1),
            'ID': all_proteome_ids.astype(int),
        })
        df.to_csv('predictions.csv', index=False)
        self.print("Predictions saved to 'predictions.csv'.")



