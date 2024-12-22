import torch
import logging

from omegaconf import OmegaConf
import pandas as pd
from pytorch_lightning import LightningDataModule
from tdc.benchmark_group import dti_dg_group
from torch.utils.data import WeightedRandomSampler

from src.data import get_task_dir

class DGDataModule(LightningDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.logger = logging.getLogger("lightning.pytorch")
        self.prepare_data_per_node = False
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
        self._label_column="Y"
        self._seed = config.replicate
        self.classify = config.classify
        self.bins = config.bins
        self.dataset_path = config.dataset_path
        self.load_data()

    def load_data(self):
        self._dg_group = dti_dg_group(path=self._data_dir)
        self._dg_data = self._dg_group.get("bindingdb_patent")
        self._train_val, self.df_test = (
            self._dg_data["train_val"],
            self._dg_data["test"],
        )
        self._df = pd.concat([self._train_val, self.df_test])

    def prepare_featurizer(self, featurizer, all_items):
        if featurizer.path.exists():
            self.logger.warning("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            featurizer.cuda(self._device)
        featurizer.write_to_disk(all_items)
        featurizer.cpu()

    def calculate_weights(self, label_dict, dataset,lable_column):
        class_weights = {}
        total_samples = len(dataset)
        for label, count in label_dict.items():
            if count == 0:
                weight = 0
            else:
                weight = total_samples / count
            class_weights[label] = weight
        arr = []
        for index, row in dataset.iterrows():
            weight = class_weights[row[lable_column]]
            arr.append(weight)
        return arr

    def build_weighted_sampler(self, dataset, lable_column):
        label_dict = dataset[lable_column].value_counts().to_dict()
        weights = self.calculate_weights(label_dict, dataset,lable_column)
        weights = torch.DoubleTensor(weights)
        return WeightedRandomSampler(weights, len(dataset), replacement=True)

    @property
    def all_targets(self):
        return self._df[self._target_column].unique()

    @property
    def all_drugs(self):
        return self._df[self._drug_column].unique()

    def prepare_data(self):
        raise NotImplemented

    def setup_featurizer(self, featurizer, all_items):
        featurizer.preload(all_items)
        featurizer.cpu()

    def setup(self, stage: str):
        raise NotImplemented

    def train_dataloader(self):
        raise NotImplemented

    def val_dataloader(self):
        raise NotImplemented

    def test_dataloader(self):
        raise NotImplemented
