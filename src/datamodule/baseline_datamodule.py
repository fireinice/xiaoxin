import numpy as np
import pandas as pd
import torch
import logging
import os
import hashlib

from omegaconf import OmegaConf
import typing as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from src.utils import get_featurizer, logg
from src.datamodule.dg_datamodule import DGDataModule
from src.datamodule.pre_encoded_datamodule import BinaryDataset



def filter_max_segment(group):
    category_counts = group['Y'].value_counts()
    max_count = category_counts.max()
    max_categories = category_counts[category_counts == max_count].index
    if len(max_categories) > 1:
        return pd.DataFrame(columns=group.columns)
    max_category = max_categories[0]
    return group[group['Y'] == max_category]

def get_cache_path(base_path, bins, dataset_name):
    bins_str = "_".join(map(str, bins))
    bins_hash = hashlib.md5(bins_str.encode()).hexdigest()
    cache_file = os.path.join(base_path, f"cache_{dataset_name}_{bins_hash}.csv")
    return cache_file


def subsection(df, bins: list, is_train_val: bool, dataset_name,base_path="."):
    cache_path = get_cache_path(base_path, bins, dataset_name)

    if os.path.exists(cache_path):
        print(f"加载缓存文件: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"未找到缓存文件，重新计算: {cache_path}")
    df['Combine'] = df['Drug'] + '_' + df['Target']
    bins.append(float(np.inf))
    labels = list(range(len(bins) - 1))
    df['Y'] = pd.cut(df['Y'], bins=bins, labels=labels, right=False)

    if is_train_val:
        if all(df.groupby('Combine')['Y'].nunique() == 1):
            print("所有组合已在单一段，无需筛选。")
        else:
            df = df.groupby('Combine', group_keys=False).apply(filter_max_segment)
            print("筛选完成。")
    else:
        print("不需要筛选")

    df = df.drop(columns='Combine')
    bins.pop()

    df.to_csv(cache_path, index=False)
    print(f"结果已缓存至: {cache_path}")
    return df

def regression(df,dataset_name, base_path="."):
    cache_path = get_cache_path(base_path, ["regression"],dataset_name)
    if os.path.exists(cache_path):
        print(f"加载缓存文件: {cache_path}")
        return pd.read_csv(cache_path)
    print(f"未找到缓存文件，重新计算: {cache_path}")
    df['Y'] = np.log(df['Y'])
    df.to_csv(cache_path, index=False)
    print(f"结果已缓存至: {cache_path}")
    return df


class BaselineDataModule(DGDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("BaselineDataModule")
        if config.model_architecture in (
            "MorganAttention",
        ):
            self.attention = True
        else:
            self.attention = False

        self.drug_featurizer = get_featurizer(
            config.drug_featurizer, save_dir=self._task_dir
        )
        self.target_featurizer = get_featurizer(
            config.target_featurizer, per_tok=self.attention, save_dir=self._task_dir
        )
        self.weights = None

    def prepare_data(self):
        self.prepare_featurizer(self.target_featurizer, self.all_targets)
        self.prepare_featurizer(self.drug_featurizer, self.all_drugs)

    def process_data(self):
        dg_name = self._dg_data["name"]
        self.df_train, self.df_val = self._dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="random", seed=self._seed
        )

        if self.classify:
            self.df_train = subsection(self.df_train, self.bins, True, 'train', self._data_dir)
            self.df_val = subsection(self.df_val, self.bins, True, 'val', self._data_dir)
            self.df_test = subsection(self.df_test, self.bins, False, 'test', self._data_dir)
        else:
            self.df_train = regression(self.df_train, 'train', self._data_dir)
            self.df_val = regression(self.df_val, 'val', self._data_dir)
            self.df_test = regression(self.df_test, 'test', self._data_dir)


    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.process_data()
        self.sampler = self.build_weighted_sampler(self.df_train,self._label_column)

        if stage == "fit" or stage is None:
            self.train_data = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )
            self.val_data = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )
            self.test_data = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.test_data = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )


    def _collate_fn(self, args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        d_emb = [a[0] for a in args]
        t_emb = [a[1] for a in args]
        labs = [a[2] for a in args]

        try:
            drugs = torch.stack(d_emb, 0)
        except Exception as e:
            logg.error(f"Testing failed with exception {e}")
            print(d_emb)

        targets = pad_sequence(
            t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        labels = torch.stack(labs, 0)

        return drugs, targets, labels.to(torch.int64)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            # shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            pin_memory=True,
            sampler=self.sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            pin_memory=True
        )
