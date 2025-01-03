import logging

import pandas as pd
import torch
from omegaconf import OmegaConf

import typing as T
from torch.nn.utils.rnn import pad_sequence

from src.datamodule.bacteria_predict_datamodule import BacteriaPredictDataModule
from src.datamodule.morgan_chembert_datamodule import BinaryDataset_Double, MorganChembertDataModule
from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from torch.utils.data import Dataset, DataLoader


class MorganChembertPredictDataModule(MorganChembertDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("BacteriaDataModule")
        self._index = "Index"

    def load_data(self):
        total_rows = sum(1 for line in open(self.dataset_path))
        half_rows = total_rows // 2
        # self._df = pd.read_csv(self.dataset_path, nrows=half_rows)
        self._df = pd.read_csv(self.dataset_path, skiprows=range(1, half_rows + 1))

    def prepare_data(self):
        super(MorganChembertPredictDataModule, self).prepare_data()

    def setup(self,stage:str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.setup_featurizer(self.drug_featurizer_two,self.all_drugs)
        if stage == 'predict' or stage is None:
            self.predict_data = BinaryDataset_Double(
                    self._df[self._drug_column],
                    self._df[self._target_column],
                    self._df[self._index],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )

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
