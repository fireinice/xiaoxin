import typing as T

import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.featurizers.molecule import ChemBertaFeaturizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX, ProtBertFeaturizer
from src.featurizers import Featurizer
from src.datamodule.finetune_chembert_datamodule import FineTuneChemBertDataModule


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


class PreEncodedDataModule(FineTuneChemBertDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self._loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "collate_fn": self.test_collate_fn,
        }
        self.drug_featurizer = ChemBertaFeaturizer(save_dir=self._task_dir)
        self.target_featurizer = ProtBertFeaturizer(
            per_tok=self.cross_attention, save_dir=self._task_dir
        )

    def prepare_data(self):
        self.prepare_featurizer(self.target_featurizer, self.all_targets)
        self.prepare_featurizer(self.drug_featurizer, self.all_drugs)

    def setup(self, stage):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        dg_name = self._dg_data["name"]
        self.df_train, self.df_val = self._dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="random", seed=self._seed
        )

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

        if stage == "test" or stage is None:
            self.test_data = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def test_collate_fn(self, args: T.Tuple[Tensor, Tensor, Tensor]):
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
        drugs = pad_sequence(
            d_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )
        targets = pad_sequence(
            t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )
        labels = torch.stack(labs, 0)
        return drugs, targets, labels

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, drop_last=True)
        return DataLoader(self.data_val, **self._loader_kwargs, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs, drop_last=True)
