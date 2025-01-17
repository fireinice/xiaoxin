import torch
import logging
from omegaconf import OmegaConf
import typing as T
from torch.nn.utils.rnn import pad_sequence
from src.datamodule.baseline_datamodule import BaselineDataModule, BinaryDataset
from src.featurizers import Featurizer
from src.featurizers.protein import FOLDSEEK_MISSING_IDX

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


class MorganChembertaDataModule(BaselineDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("MorganChembertDataModule")
        self.drug_featurizer_two = self.drug_featurizer[1]
        self.drug_featurizer = self.drug_featurizer[0]

    def prepare_data(self):
        super(MorganChembertaDataModule, self).prepare_data()
        self.prepare_featurizer(self.drug_featurizer_two,self.all_drugs)

    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.setup_featurizer(self.drug_featurizer_two,self.all_drugs)
        if stage in ['fit', 'validate', 'test']:
            self.process_data()
            if stage == "fit":
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
                self.test_data = BinaryDataset_Double(
                    self.df_test[self._drug_column],
                    self.df_test[self._target_column],
                    self.df_test[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
            if stage == "test" or stage == "validate":
                self.test_data = BinaryDataset_Double(
                    self.df_test[self._drug_column],
                    self.df_test[self._target_column],
                    self.df_test[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
        if stage == "predict":
            self.predict_data = BinaryDataset_Double(
                    self._df[self._drug_column],
                    self._df[self._target_column],
                    self._df[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )


    def _collate_fn(
        self,args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        d_emb_one = [a[0] for a in args]
        d_emb_two = [a[1] for a in args]
        t_emb = [a[2] for a in args]
        labs = [a[3] for a in args]
        drugs_one = torch.stack(d_emb_one, 0)
        drugs_two = torch.stack(d_emb_two, 0)
        drugs = {
            "drugs_one": drugs_one,
            "drugs_two": drugs_two,
        }
        targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
        labels = torch.stack(labs, 0)

        return drugs, targets, labels



