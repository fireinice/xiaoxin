import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dscript
import os
import sys
import pickle as pk
import pandas as pd
import pytorch_lightning as pl

from functools import partial

from types import SimpleNamespace
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import lru_cache
from numpy.random import choice
from sklearn.model_selection import KFold, train_test_split
from torch.nn.utils.rnn import pad_sequence
from tdc.benchmark_group import dti_dg_group

from .featurizers import Featurizer
from .featurizers.protein import FOLDSEEK_MISSING_IDX
from .utils import get_logger
from pathlib import Path
import typing as T
import pandas as pd
import numpy as np

logg = get_logger()


def get_task_dir(task_name: str):
    """
    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    task_paths = {
        "biosnap": "./dataset/BIOSNAP/full_data",
        "biosnap_prot": "./dataset/BIOSNAP/unseen_protein",
        "biosnap_mol": "./dataset/BIOSNAP/unseen_drug",
        "bindingdb": "./dataset/BindingDB",
        "davis": "./dataset/DAVIS",
        "dti_dg": "./dataset/TDC",
        "dude": "./dataset/DUDe",
        "halogenase": "./dataset/EnzPred/halogenase_NaCl_binary",
        "bkace": "./dataset/EnzPred/duf_binary",
        "gt": "./dataset/EnzPred/gt_acceptors_achiral_binary",
        "esterase": "./dataset/EnzPred/esterase_binary",
        "kinase": "./dataset/EnzPred/davis_filtered",
        "phosphatase": "./dataset/EnzPred/phosphatase_chiral_binary",
        "bindingdb_v2": "./dataset/BingdingDB_v2",
        "bindingdb_multi_class": "./dataset/BindingDB_multi_class",
        "bindingdb_multi_class_small": "./dataset/BindingDB_multi_class_small",
        "pcctomuilt": "./dataset/pcctomuilt"
    }

    return Path(task_paths[task_name.lower()]).resolve()


def contrastive_collate_fn(
        args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    Specific collate function for contrastive dataloader

    :param args: Batch of training samples with anchor, positive, negative
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    anchor_emb = [a[0] for a in args]
    pos_emb = [a[1] for a in args]
    neg_emb = [a[2] for a in args]

    anchors = pad_sequence(
        anchor_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
    )
    positives = torch.stack(pos_emb, 0)
    negatives = torch.stack(neg_emb, 0)

    return anchors, positives, negatives


def make_contrastive(
        df: pd.DataFrame,
        posneg_column: str,
        anchor_column: str,
        label_column: str,
        n_neg_per: int = 50,
):
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]

    contrastive = []

    for _, r in pos_df.iterrows():
        for _ in range(n_neg_per):
            contrastive.append(
                (
                    r[anchor_column],
                    r[posneg_column],
                    choice(neg_df[posneg_column]),
                )
            )

    contrastive = pd.DataFrame(
        contrastive, columns=["Anchor", "Positive", "Negative"]
    )
    return contrastive

def drug_target_collate_fn(
        args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
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

    return drugs, targets, labels

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

class ContrastiveDataset(Dataset):
    def __init__(
            self,
            anchors,
            positives,
            negatives,
            posneg_featurizer: Featurizer,
            anchor_featurizer: Featurizer,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        self.posneg_featurizer = posneg_featurizer
        self.anchor_featurizer = anchor_featurizer

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        anchorEmb = self.anchor_featurizer(self.anchors[i])
        positiveEmb = self.posneg_featurizer(self.positives[i])
        negativeEmb = self.posneg_featurizer(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb

class DTIDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")

        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):

        if (
                self.drug_featurizer.path.exists()
                and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )

        df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )

        df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )

        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat(
            [i[self._drug_column] for i in dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):

        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )

        self.df_val = pd.read_csv(
            self._data_dir / self._val_path, **self._csv_kwargs
        )

        self.df_test = pd.read_csv(
            self._data_dir / self._test_path, **self._csv_kwargs
        )

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class EnzPredDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_file = Path(data_dir).with_suffix(".csv")
        self._data_stem = Path(self._data_file.stem)
        self._data_dir = self._data_file.parent / self._data_file.stem
        self._seed = 0
        self._replicate = seed

        df = pd.read_csv(self._data_file, index_col=0)
        self._drug_column = df.columns[1]
        self._target_column = df.columns[0]
        self._label_column = df.columns[2]

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    @classmethod
    def dataset_list(cls):
        return [
            "halogenase",
            "bkace",
            "gt",
            "esterase",
            "kinase",
            "phosphatase",
        ]

    def prepare_data(self):

        os.makedirs(self._data_dir, exist_ok=True)

        kfsplitter = KFold(n_splits=10, shuffle=True, random_state=self._seed)
        full_data = pd.read_csv(self._data_file, index_col=0)

        all_drugs = full_data[self._drug_column].unique()
        all_targets = full_data[self._target_column].unique()

        if (
                self.drug_featurizer.path.exists()
                and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug and target featurizers already exist")

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

        for i, split in enumerate(kfsplitter.split(full_data)):
            fold_train = full_data.iloc[split[0]].reset_index(drop=True)
            fold_test = full_data.iloc[split[1]].reset_index(drop=True)
            logg.debug(
                self._data_dir / self._data_stem.with_suffix(f".{i}.train.csv")
            )
            fold_train.to_csv(
                self._data_dir
                / self._data_stem.with_suffix(f".{i}.train.csv"),
                index=True,
                header=True,
            )
            fold_test.to_csv(
                self._data_dir / self._data_stem.with_suffix(f".{i}.test.csv"),
                index=True,
                header=True,
            )

    def setup(self, stage: T.Optional[str] = None):

        df_train = pd.read_csv(
            self._data_dir
            / self._data_stem.with_suffix(f".{self._replicate}.train.csv"),
            index_col=0,
        )
        self.df_train, self.df_val = train_test_split(df_train, test_size=0.1)
        self.df_test = pd.read_csv(
            self._data_dir
            / self._data_stem.with_suffix(f".{self._replicate}.test.csv"),
            index_col=0,
        )

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class DUDEDataModule(pl.LightningDataModule):
    def __init__(
            self,
            contrastive_split: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            device: torch.device = torch.device("cpu"),
            n_neg_per: int = 50,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=None,
            sep="\t",
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": contrastive_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device
        self._n_neg_per = n_neg_per

        self._data_dir = Path("./dataset/DUDe/")
        self._split = contrastive_split
        self._split_path = self._data_dir / Path(
            f"dude_{self._split}_type_train_test_split.csv"
        )

        self._drug_id_column = "Molecule_ID"
        self._drug_column = "Molecule_SMILES"
        self._target_id_column = "Target_ID"
        self._target_column = "Target_Seq"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        pass

    #         self.df_full = pd.read_csv(
    #             self._data_dir / Path("full.tsv"), **self._csv_kwargs
    #         )
    #         all_drugs = self.df_full[self._drug_column].unique()
    #         all_targets = self.df_full[self._target_column].unique()

    #         if self._device.type == "cuda":
    #             self.drug_featurizer.cuda(self._device)
    #             self.target_featurizer.cuda(self._device)

    #         self.drug_featurizer.write_to_disk(all_drugs)
    #         self.target_featurizer.write_to_disk(all_targets)

    def setup(self, stage: T.Optional[str] = None):

        self.df_full = pd.read_csv(
            self._data_dir / Path("full.tsv"), **self._csv_kwargs
        )

        self.df_splits = pd.read_csv(self._split_path, header=None)
        self._train_list = self.df_splits[self.df_splits[1] == "train"][
            0
        ].values
        self._test_list = self.df_splits[self.df_splits[1] == "test"][0].values

        self.df_train = self.df_full[
            self.df_full[self._target_id_column].isin(self._train_list)
        ]
        self.df_test = self.df_full[
            self.df_full[self._target_id_column].isin(self._test_list)
        ]

        self.train_contrastive = make_contrastive(
            self.df_train,
            self._drug_column,
            self._target_column,
            self._label_column,
            self._n_neg_per,
        )

        self._dataframes = [self.df_train]  # , self.df_test]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs, write_first=True)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets, write_first=True)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = ContrastiveDataset(
                self.train_contrastive["Anchor"],
                self.train_contrastive["Positive"],
                self.train_contrastive["Negative"],
                self.drug_featurizer,
                self.target_featurizer,
            )

        # if stage == "test" or stage is None:
        #     self.data_test = BinaryDataset(self.df_test[self._drug_column],
        #                                     self.df_test[self._target_column],
        #                                     self.df_test[self._label_column],
        #                                     self.drug_featurizer,
        #                                     self.target_featurizer
        #                                    )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)


#     def val_dataloader(self):
#         return DataLoader(self.data_test,
#                         **self._loader_kwargs
#                          )

#     def test_dataloader(self):
#         return DataLoader(self.data_test,
#                          **self._loader_kwargs
#                          )

def filter_max_segment(group):
    category_counts = group['Y'].value_counts()
    max_count = category_counts.max()
    max_categories = category_counts[category_counts == max_count].index
    if len(max_categories) > 1:
        return pd.DataFrame(columns=group.columns)
    max_category = max_categories[0]
    return group[group['Y'] == max_category]

def subsection(df,bins):
    df['Combine'] = df['Drug'] + '_' + df['Target']
    bins.append(float(np.inf))
    labels = list(range(len(bins) - 1))
    df['Y'] = pd.cut(df['Y'], bins=bins, labels=labels, right=False)
    if all(df.groupby('Combine')['Y'].nunique() == 1):
        print("所有组合已在单一段，无需筛选。")
    else:
        df = df.groupby('Combine', group_keys=False).apply(filter_max_segment)
        print("筛选完成。")
    df = df.drop(columns='Combine')
    bins.pop()
    return df

def regression(df):
    df['Y'] = np.log(df['Y'])
    return df

#基类
class TDCDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            bins: list,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
            classify = False
    ):

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._seed = seed

        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"
        self._classify = classify
        self._bins = bins

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):

        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")

        train_val, test = (
            dg_benchmark["train_val"],
            dg_benchmark["test"],
        )

        self.all_drugs = pd.concat([train_val, test])[self._drug_column].unique()
        self.all_targets = pd.concat([train_val, test])[
            self._target_column
        ].unique()

        if (
                self.drug_featurizer.path.exists()
                and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(self.all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(self.all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):

        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")
        dg_name = dg_benchmark["name"]

        self.df_train, self.df_val = dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="random", seed=self._seed
        )
        self.df_test = dg_benchmark["test"]

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        if self._classify:
            self._dataframes = [subsection(i,bins=self._bins) for i in self._dataframes]
        else:
            self._dataframes = [regression(i) for i in self._dataframes]

        all_drugs = pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, drop_last=True)
#直接读取CSV文件
class CustomDataset(Dataset):
    def __init__(self, dataframe, indices):
        self.dataframe = dataframe
        self.indices = indices
        self.token_cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataframe.loc[real_idx]

        drug = item['Drug']
        target = item['Target']
        label = item['Y']
        return drug, target, label

class CSVDataModule(TDCDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer,
            target_featurizer,
            bins: list,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            classify = False
    ):
        # 调用父类的初始化方法
        super().__init__(
            data_dir=data_dir,
            drug_featurizer=drug_featurizer,
            target_featurizer=target_featurizer,
            device=device,
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            classify = classify,
            bins=bins
        )

        # 初始化数据集相关变量
        self.token_cache = {}

    def prepare_data(self):
        # 加载训练、验证和测试数据
        self.train_val_data = pd.read_csv(self.data_dir / "train_val.csv")
        self.test_data = pd.read_csv(self.data_dir / "test.csv")

        # 清理数据，去掉缺失值的行
        self.train_val_data.dropna(subset=["Drug", "Target", self.label_column], inplace=True)
        self.test_data.dropna(subset=["Drug", "Target", self.label_column], inplace=True)

        if self._classify:
            self.train_val_data = subsection(self.train_val_data,self._bins)
            self.test_data = subsection(self.test_data,self._bins)
        else:
            self.train_val_data = regression(self.train_val_data)
            self.test_data = regression(self.test_data)

    def setup(self, stage: T.Optional[str] = None):
        import numpy as np

        # 划分训练集和验证集
        if stage == "fit" or stage is None:
            df_indices = self.train_val_data.index.tolist()

            np.random.seed(self.seed)
            np.random.shuffle(df_indices)

            train_size = int(0.8 * len(df_indices))  # 80% 训练集，20% 验证集
            train_indices = df_indices[:train_size]
            val_indices = df_indices[train_size:]

            self.train_data = CustomDataset(self.train_val_data, train_indices)
            self.val_data = CustomDataset(self.train_val_data, val_indices)

        # 测试集数据
        if stage == "test" or stage is None:
            test_indices = self.test_data.index.tolist()
            self.test_data = CustomDataset(self.test_data, test_indices)

    def _collate_fn(self, drug_featurizer, target_featurizer, token_cache):
        def _fn(batch):
            drugs = []
            targets = []
            labels = []

            for item in batch:
                drug, target, label = item
                drugs.append(drug)

                # 使用缓存来避免重复计算靶标嵌入
                if target not in token_cache:
                    target_embedding = target_featurizer._tokenizer(target)
                    token_cache[target] = target_embedding
                else:
                    target_embedding = token_cache[target]

                targets.append(target_embedding)
                labels.append(label)

            # 对药物进行token化
            drug_tokens = drug_featurizer._tokenizer(drugs)

            # 生成药物的输入数据
            new_drug_tokens = {
                'drug_input_ids': drug_tokens['input_ids'],
                'drug_att_masks': drug_tokens['attention_mask']
            }

            # 对靶标进行填充，使得批次大小一致
            targets = pad_sequence(targets, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)

            # 将标签转换为torch张量
            labels = torch.tensor(np.array(labels))

            return new_drug_tokens, targets, labels

        return _fn

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, collate_fn=self._collate_fn(self.drug_featurizer, self.target_featurizer, self.token_cache))

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self._collate_fn(self.drug_featurizer, self.target_featurizer, self.token_cache))

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self._collate_fn(self.drug_featurizer, self.target_featurizer, self.token_cache))
#使用缓存的三维的chemberta
class TDCDataModule_Local(TDCDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: Featurizer,
            target_featurizer: Featurizer,
            bins:list,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
            classify=False
    ):
        # 调用父类的初始化方法
        super().__init__(
            data_dir=data_dir,
            drug_featurizer=drug_featurizer,
            target_featurizer=target_featurizer,
            device=device,
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            classify = classify,
            bins=bins
        )

        # 数据加载参数
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn":self._collate_fn,  # 使用自定义的collate_fn
        }

    def _collate_fn(
         self, args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        d_emb = [a[0] for a in args]
        t_emb = [a[1] for a in args]
        labs = [a[2] for a in args]

        drugs = pad_sequence(
            d_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        targets = pad_sequence(
            t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        labels = torch.stack(labs, 0)

        return drugs, targets, labels

    def prepare_data(self):
        super().prepare_data()
    def setup(self, stage: T.Optional[str] = None):
        super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, drop_last=True)

#同时使用Morgan和Chemberta
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
        # 调用父类的初始化方法，初始化药物、靶标和标签
        super().__init__(drugs, targets, labels, drug_featurizer_one, target_featurizer)

        # 初始化第二个药物特征化器
        self.drug_featurizer_two = drug_featurizer_two

    def __getitem__(self, i: int):
        # 获取第一个药物的特征
        drug_one = self.drug_featurizer(self.drugs.iloc[i])
        # 获取第二个药物的特征
        drug_two = self.drug_featurizer_two(self.drugs.iloc[i])

        # 获取靶标的特征
        target = self.target_featurizer(self.targets.iloc[i])

        # 获取标签
        label = torch.tensor(self.labels.iloc[i])

        return drug_one, drug_two, target, label

class TDCDataModule_Double(TDCDataModule):
    def __init__(
            self,
            data_dir: str,
            drug_featurizer: list,  # 传入两个药物特征化器
            target_featurizer: Featurizer,
            bins: list,
            device: torch.device = torch.device("cpu"),
            seed: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            header=0,
            index_col=0,
            sep=",",
            classify = False
    ):
        # 调用父类的初始化方法
        super().__init__(
            data_dir=data_dir,
            drug_featurizer=drug_featurizer[0],  # 使用第一个药物特征化器初始化
            target_featurizer=target_featurizer,
            device=device,
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            header=header,
            index_col=index_col,
            sep=sep,
            classify=classify,
            bins=bins
        )

        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": self._collate_fn # 使用自定义的collate_fn
        }

        # 初始化第二个药物特征化器
        self.drug_featurizer_two = drug_featurizer[1]

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

        return drugs, targets, labels

    def prepare_data(self):
        # 准备数据，与父类保持一致
        super().prepare_data()
        if (
            self.drug_featurizer.path.exists()
            and self.drug_featurizer_two.path.exists()
            and self.target_featurizer.path.exists()
        ):
            logg.warning("Drug_one and Drug_two and target featurizers already exist")
            return

        if self._device.type == "cuda":
            self.drug_featurizer_two.cuda(self._device)

        if not self.drug_featurizer_two.path.exists():
            self.drug_featurizer_two.write_to_disk(self.all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(self.all_targets)

        self.drug_featurizer_two.cpu()

    def setup(self, stage: T.Optional[str] = None):
        # 设置训练/验证/测试数据
        super().setup(stage)

        # 添加第二个药物特征化器的预处理
        if self._device.type == "cuda":
            self.drug_featurizer_two.cuda(self._device)

        self.drug_featurizer_two.preload(pd.concat(
            [i[self._drug_column] for i in self._dataframes]
        ).unique())

        self.drug_featurizer_two.cpu()

        # 在训练/验证/测试数据集初始化时加入第二个药物特征化器
        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset_Double(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset_Double(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset_Double(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.drug_featurizer_two,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, drop_last=True)