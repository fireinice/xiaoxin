import ast

import numpy as np
import torch
import logging

from omegaconf import OmegaConf
from typing import Tuple, List
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.datamodule.baseline_datamodule import subsection, regression, BaselineDataModule
from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from src.datamodule.pre_encoded_datamodule import BinaryDataset


class BinaryDataset_Bacteria(BinaryDataset):
    def __init__(
            self,
            drugs,
            targets:list,
            labels,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
    ):
        super().__init__(drugs, targets, labels, drug_featurizer, target_featurizer)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])

        target_features = [self.target_featurizer(target) for target in ast.literal_eval(self.targets.iloc[i])]
        target = torch.stack(target_features, dim=0)

        label = torch.tensor(self.labels.iloc[i])

        return drug, target, label

class BacteriaDataModule(BaselineDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("BacteriaDataModule")

    @property
    def all_targets(self):
        all_target_sequences = []
        for target in self._df[self._target_column]:
            target_list = ast.literal_eval(target)
            all_target_sequences.extend(target_list)
        all_target_sequences = list(dict.fromkeys(all_target_sequences))
        return all_target_sequences

    def prepare_data(self):
        super(BacteriaDataModule, self).prepare_data()

    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.process_data()
        self.sampler = self.build_weighted_sampler(self.df_train, self._label_column)

        if stage == "fit" or stage is None:
            self.train_data = BinaryDataset_Bacteria(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )
            self.val_data = BinaryDataset_Bacteria(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )
            self.test_data = BinaryDataset_Bacteria(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.test_data = BinaryDataset_Bacteria(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
            # sampler=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True
        )


