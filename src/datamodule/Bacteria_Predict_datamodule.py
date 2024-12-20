import logging
from argparse import ArgumentParser

import pandas as pd
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
import ast

import typing as T
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.datamodule.Bacteria_datamodule import BacteriaDataModule, BinaryDataset_Bacteria
from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from src.models.morgan_chembert_model import MorganChembertAttention
from torch.utils.data import Dataset, DataLoader
from src.utils import logg


class BinaryDataset_predict(Dataset):
    def __init__(
            self,
            drugs,
            targets:list,
            Proteome_ID,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
    ):
        self.drugs = drugs
        self.targets = targets
        self.Proteome_ID = Proteome_ID
        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        Proteome_ID = torch.tensor(int(self.Proteome_ID.iloc[i].lstrip('UP')))
        target_features = [self.target_featurizer(target) for target in ast.literal_eval(self.targets.iloc[i])]
        target = torch.stack(target_features, dim=0)
        return drug, target, Proteome_ID


class BacteriaPredictDataModule(BacteriaDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("BacteriaDataModule")
        self._Proteome_ID = 'Proteome_ID'

    def load_data(self):
        self._df = pd.read_csv(self.dataset_path)

    def prepare_data(self):
        super(BacteriaPredictDataModule, self).prepare_data()

    def setup(self,stage:str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        if stage == 'predict' or stage is None:
            self.predict_data = BinaryDataset_predict(
                    self._df[self._drug_column],
                    self._df[self._target_column],
                    self._df[self._Proteome_ID],
                    self.drug_featurizer,
                    self.target_featurizer,
                )

    def _collate_fn(self, args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        d_emb = [a[0] for a in args]
        t_emb = [a[1] for a in args]
        ids = [a[2] for a in args]

        try:
            drugs = torch.stack(d_emb, 0)
        except Exception as e:
            logg.error(f"Testing failed with exception {e}")
            print(d_emb)

        targets = pad_sequence(
            t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        ids = torch.stack(ids, 0)

        return drugs, targets, ids

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True
        )
