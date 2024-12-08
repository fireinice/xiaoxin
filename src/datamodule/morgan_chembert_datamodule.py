import torch
import logging

from omegaconf import OmegaConf

import typing as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.datamodule.baseline_datamodule import subsection, regression, BaselineDataModule
from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from src.utils import get_featurizer, logg
from src.datamodule.dg_datamodule import DGDataModule
from src.datamodule.pre_encoded_datamodule import BinaryDataset

class BinaryDataset_Double(BinaryDataset):
    def __init__(
            self,
            drugs,
            targets,
            labels,
            drug_featurizer_one: Featurizer,
            drug_featurizer_two: Featurizer,
            target_featurizer: Featurizer,
    ):
        super().__init__(drugs, targets, labels, drug_featurizer_one, target_featurizer)

        self.drug_featurizer_two = drug_featurizer_two

    def __getitem__(self, i: int):
        drug_one = self.drug_featurizer(self.drugs.iloc[i])
        drug_two = self.drug_featurizer_two(self.drugs.iloc[i])
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i])
        return drug_one, drug_two, target, label


class MorganChembertDataModule(BaselineDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("MorganChembertDataModule")
        if config.model_architecture in (
            "MorganChembertAttention",
        ):
            self.attention = True
        else:
            self.attention = False

        self.drug_featurizer_two = self.drug_featurizer[1]
        self.drug_featurizer = self.drug_featurizer[0]
    def prepare_data(self):
        super(MorganChembertDataModule, self).prepare_data()
        self.prepare_featurizer(self.drug_featurizer_two,self.all_drugs)

    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.setup_featurizer(self.drug_featurizer_two,self.all_drugs)
        dg_name = self._dg_data["name"]
        self.df_train, self.df_val = self._dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="random", seed=self._seed
        )

        if self.classify:
            self.df_train = subsection(self.df_train, self.bins, True,'train',self._data_dir)
            self.df_val = subsection(self.df_val, self.bins, True,'val',self._data_dir)
            self.df_test = subsection(self.df_test, self.bins, False,'test',self._data_dir)
        else:
            self.df_train = regression(self.df_train, 'train',self._data_dir)
            self.df_val = regression(self.df_val, 'val',self._data_dir)
            self.df_test = regression(self.df_test, 'test',self._data_dir)

        label_counts = self.df_train[self._label_column].value_counts().to_dict()
        weights = self.calculate_weights(label_counts, self.df_train)
        self._weights = torch.DoubleTensor(weights)

        self.sampler = WeightedRandomSampler(
            self._weights, len(self.df_train), replacement=True
        )

        if stage == "fit" or stage is None:
            self.train_data = BinaryDataset_Double(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,
            )

            self.val_data = BinaryDataset_Double(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.test_data = BinaryDataset_Double(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,)

        if stage == "test" or stage is None:
            self.test_data = BinaryDataset_Double(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,)

    def calculate_weights(self, label_dict, dataset):
        arr = []
        for label, count in label_dict.items():
            weight = len(dataset) / count
            arr.append(weight)
        return arr

    def _collate_fn(
        self,args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        d_emb_one = [a[0] for a in args]
        d_emb_two = [a[1] for a in args]
        t_emb = [a[2] for a in args]
        labs = [a[3] for a in args]

        # 堆叠药物特征
        drugs_one = torch.stack(d_emb_one, 0)
        drugs_two = torch.stack(d_emb_two, 0)

        drugs = {
            "drugs_one": drugs_one,
            "drugs_two": drugs_two,
        }

        # 对靶标进行填充
        targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)

        labels = torch.stack(labs, 0)

        return drugs, targets, labels.to(torch.int64)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
            sampler=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )


