import torch
import logging

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from src.data import TDCDataModule_Double, get_task_dir
from src.utils import get_featurizer

class MorganChembertDataModule(LightningDataModule):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__()
        self.logger = logging.getLogger("MorganChembertDataModule")
        self.prepare_data_per_node = False
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        task_dir = get_task_dir(config.task)
        if len(config.ds) > 0:
            suffix_task_dir = task_dir.with_suffix(f".{config.ds}")
            if suffix_task_dir.is_dir():
                task_dir = suffix_task_dir
            else:
                self.logger.warn(f"Cannot find dataset {str(suffix_task_dir)}, fall back to {task_dir}")
        drug_featurizer = get_featurizer(
            config.drug_featurizer, per_tok=False, save_dir=task_dir
        )
        target_featurizer = get_featurizer(
            config.target_featurizer, per_tok=True, save_dir=task_dir
        )

        self.datamodule = TDCDataModule_Double(
            task_dir,
            drug_featurizer,
            target_featurizer,
            device=device,
            seed=config.replicate,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            label_column=config.label_column
        )

    def prepare_data(self):
        self.datamodule.prepare_data()

    def setup(self, stage: str):
        self.datamodule.setup()

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.test_dataloader()
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
