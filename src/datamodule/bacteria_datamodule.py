import logging
from omegaconf import OmegaConf
import ast
import torch
from src.datamodule.morgan_chemberta_datamodule import BinaryDataset_Double, MorganChembertaDataModule
from src.featurizers import Featurizer


class BinaryDatasetBiFeatures(BinaryDataset_Double):
    def __init__(
            self,
            drugs,
            targets: list,
            labels,
            drug_featurizer_one: Featurizer,
            drug_featurizer_two: Featurizer,
            target_featurizer: Featurizer,
    ):
        super().__init__(drugs, targets, labels, drug_featurizer_one, drug_featurizer_two,target_featurizer)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target_features = [self.target_featurizer(target) for target in ast.literal_eval(self.targets.iloc[i])]
        target = torch.stack(target_features, dim=0)

        if type(self.labels.iloc[i])==str:
            label = torch.tensor(int(self.labels.iloc[i].lstrip('UP')))
        else:
            label = torch.tensor(self.labels.iloc[i])

        return drug, target, label

class BacteriaDataModule(MorganChembertaDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        self.logger = logging.getLogger("BacteriaDataModule")
        self._all_target_sequences_cache = None

    @property
    def all_targets(self):
        if self._all_target_sequences_cache is not None:
            return self._all_target_sequences_cache
        all_target_sequences = []
        for target in self._df[self._target_column]:
            target_list = ast.literal_eval(target)
            all_target_sequences.extend(target_list)
        self._all_target_sequences_cache = list(dict.fromkeys(all_target_sequences))
        return self._all_target_sequences_cache


    def prepare_data(self):
        super(BacteriaDataModule, self).prepare_data()

    def setup(self, stage: str):
        self.setup_featurizer(self.target_featurizer, self.all_targets)
        self.setup_featurizer(self.drug_featurizer, self.all_drugs)
        self.setup_featurizer(self.drug_featurizer_two,self.all_drugs)
        if stage in ['fit', 'validate', 'test']:
            self.process_data()
            if stage == "fit":
                self.train_data = BinaryDatasetBiFeatures(
                    self.df_train[self._drug_column],
                    self.df_train[self._target_column],
                    self.df_train[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
                self.val_data = BinaryDatasetBiFeatures(
                    self.df_val[self._drug_column],
                    self.df_val[self._target_column],
                    self.df_val[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
                self.test_data = BinaryDatasetBiFeatures(
                    self.df_test[self._drug_column],
                    self.df_test[self._target_column],
                    self.df_test[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
            if stage == "test" or stage == "validate":
                self.test_data = BinaryDatasetBiFeatures(
                    self.df_test[self._drug_column],
                    self.df_test[self._target_column],
                    self.df_test[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )
        if stage == "predict":
            self.predict_data = BinaryDatasetBiFeatures(
                    self._df[self._drug_column],
                    self._df[self._target_column],
                    self._df[self._label_column],
                    self.drug_featurizer,
                    self.drug_featurizer_two,
                    self.target_featurizer,
                )