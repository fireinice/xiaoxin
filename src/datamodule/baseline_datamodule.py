import numpy as np
import torch
import logging
import os
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule

from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX
from src.utils import get_featurizer, get_task_dir, logg
import typing as T


class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i])
        return drug, target, label


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


def subsection(df, bins: list, is_train_val: bool, dataset_name, base_path="."):
    cache_path = get_cache_path(base_path, bins, dataset_name)

    if os.path.exists(cache_path):
        print(f"加载缓存文件: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"未找到缓存文件，重新计算: {cache_path}")
    df['Combine'] = df['Drug'] + '_' + df['Target']
    bins.append(float(np.inf))
    labels = list(range(len(bins) - 1))
    df['Y'] = df['Y'].apply(lambda x: 1e-8 if x < 0 else x)
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

def regression(df, dataset_name, base_path="."):
    cache_path = get_cache_path(base_path, ["regression"], dataset_name)
    if os.path.exists(cache_path):
        print(f"加载缓存文件: {cache_path}")
        return pd.read_csv(cache_path)
    print(f"未找到缓存文件，重新计算: {cache_path}")
    df['Y'] = np.log(df['Y'].clip(lower=1e-8))  # 避免非正数导致的错误
    df.to_csv(cache_path, index=False)
    print(f"结果已缓存至: {cache_path}")
    return df


class BaselineDataModule(LightningDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.logger = logging.getLogger("BaselineDataModule")
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_workers = config.num_workers
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if use_cuda else "cpu")
        self._task_dir = get_task_dir(config.task)
        if len(config.ds) > 0:
            suffix_task_dir = self._task_dir.with_suffix(f".{config.ds}")
            if suffix_task_dir.is_dir():
                self._task_dir = suffix_task_dir
            else:
                self.logger.warn(
                    f"Cannot find dataset {str(suffix_task_dir)}, fall back to {self._task_dir}"
                )
        self._data_dir = self._task_dir
        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"
        self._seed = config.replicate
        self.classify = config.classify
        self.bins = config.bins
        if config.model_architecture in (
            "DrugTargetCoembeddingLightning"
        ):
            self.attention = False
        else:
            self.attention = True

        self.drug_featurizer = get_featurizer(
            config.drug_featurizer, save_dir=self._task_dir
        )
        self.target_featurizer = get_featurizer(
            config.target_featurizer, per_tok=self.attention, save_dir=self._task_dir
        )
        self.load_data()

    def load_data(self):
        all_files = [os.path.join(self._data_dir, f) for f in os.listdir(self._data_dir) if f.endswith('.csv')]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in directory: {self._data_dir}")
        train_val_file = os.path.join(self._data_dir, 'train_val.csv')
        test_file = os.path.join(self._data_dir, 'test.csv')
        preditct_file = os.path.join(self._data_dir, 'predict.csv')

        if os.path.exists(preditct_file):
            self._df = pd.read_csv(preditct_file)
        else:
            self._train_val = pd.read_csv(train_val_file)
            self.df_test = pd.read_csv(test_file)
            self._df = pd.concat([self._train_val, self.df_test])

    def prepare_featurizer(self, featurizer, all_items):
        if featurizer.path.exists():
            self.logger.warning("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            featurizer.cuda(self._device)
        featurizer.write_to_disk(all_items)
        featurizer.cpu()

    def setup_featurizer(self, featurizer, all_items):
        featurizer.preload(all_items)
        featurizer.cpu()

    @property
    def all_targets(self):
        return self._df[self._target_column].unique()

    @property
    def all_drugs(self):
        return self._df[self._drug_column].unique()

    def prepare_data(self,):
        self.prepare_featurizer(self.target_featurizer, self.all_targets)
        self.prepare_featurizer(self.drug_featurizer, self.all_drugs)

    def process_data(self):
        self.df_train, self.df_val = train_test_split(self._train_val, test_size=0.15, random_state=42)
        #bootstrap
        # self.df_test = self.df_test.sample(n=len(self.df_test), replace=True, random_state=123)
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
        if stage in ['fit', 'validate', 'test']:
            self.process_data()
            if stage == "fit":
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
            if stage == "test" or stage == "validate":
                self.test_data = BinaryDataset(
                    self.df_test[self._drug_column],
                    self.df_test[self._target_column],
                    self.df_test[self._label_column],
                    self.drug_featurizer,
                    self.target_featurizer,
                )
        if stage == "predict":
            self.predict_data = BinaryDataset(
                    self._df[self._drug_column],
                    self._df[self._target_column],
                    self._df[self._label_column],
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

        targets = pad_sequence(
            t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        labels = torch.stack(labs, 0)

        return drugs, targets, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
