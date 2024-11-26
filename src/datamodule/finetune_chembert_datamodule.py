import torch
import numpy as np
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.featurizers.protein import FOLDSEEK_MISSING_IDX, ProtBertTokenFeaturizer
from src.featurizers.molecule import ChemBertaTokenFeaturizer
from src.datamodule.dg_datamodule import DGDataModule


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        item = self.dataframe.iloc[idx]
        drug = item["Drug"]
        target = item["Target"]
        label = item["Y"]
        label = np.float32(label)
        return drug, target, label


class FineTuneChemBertDataModule(DGDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        if config.model_architecture in (
            "DrugProteinAttention",
            "DrugProteinMLP",
            "ChemBertaProteinAttention",
            "ChemBertaProteinAttention_Local",
        ):
            self.cross_attention = True
        else:
            self.cross_attention = False

        self.drug_featurizer = ChemBertaTokenFeaturizer(
            save_dir=self._task_dir
        )
        self.target_featurizer = ProtBertTokenFeaturizer(
            per_tok=self.cross_attention, save_dir=self._task_dir
        )

    def prepare_data(self):
        self.prepare_featurizer(self.target_featurizer, self.all_targets)

    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        dg_name = self._dg_data["name"]
        self.df_train, self.df_val = self._dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="random", seed=self._seed
        )

        if stage == "fit" or stage is None:
            self.train_data = CustomDataset(self.df_train)
            self.val_data = CustomDataset(self.df_val)
            self.test_data = CustomDataset(self.df_test)

        if stage == "test" or stage is None:
            self.test_data = CustomDataset(self.df_test)
        # self._dataframes = [self.df_train, self.df_val, self.df_test]
        # all_targets = pd.concat(
        # [i[self._target_column] for i in self._dataframes]

    def _collate_fn(self, batch):
        drugs = []
        targets = []
        labels = []

        for item in batch:
            drug, target, label = item
            drugs.append(drug)
            target_embedding = self.target_featurizer(target)
            targets.append(target_embedding)
            labels.append(label)

        drug_tokens = self.drug_featurizer._transfomer(drugs)

        new_drug_tokens = {}
        new_drug_tokens["drug_input_ids"] = drug_tokens["input_ids"]
        new_drug_tokens["drug_att_masks"] = drug_tokens["attention_mask"]

        targets = pad_sequence(
            targets, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
        )

        labels = torch.from_numpy(np.array(labels))

        return new_drug_tokens, targets, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
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
