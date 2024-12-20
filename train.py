import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch import nn

from src.datamodule.Bacteria_Predict_datamodule import BacteriaPredictDataModule
from src.datamodule.Bacteria_datamodule import BacteriaDataModule
from src.datamodule.baseline_datamodule import BaselineDataModule
from src.datamodule.morgan_chembert_datamodule import MorganChembertDataModule
from src.datamodule.morgan_chembert_predict_datamodule import MorganChembertPredictDataModule
from src.models.Bacteria_morgan_model import BacteriaMorganAttention
from src.models.MetricsLoss_Model import MetricsLossCallback
from src.models.lightning_model import DrugTargetCoembeddingLightning
from src.models.morgan_chembert_model import MorganChembertAttention
from src.models.morgan_model import MorganAttention

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything, strategies
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf

from src.models.drug_target_attention import DrugTargetAttention
from src.datamodule.pre_encoded_datamodule import PreEncodedDataModule
from src.datamodule.finetune_chembert_datamodule import FineTuneChemBertDataModule
from src.datamodule.morgan_chembert_datamodule import MorganChembertDataModule


def init_config() -> OmegaConf:
    parser = ArgumentParser(description="PLM_DTI Training.")
    parser.add_argument(
        "--exp-id",
        help="Experiment ID",
        dest="experiment_id",
        default="yongbo_dti_dg",
    )
    parser.add_argument(
        "--config", help="YAML config file", default="configs/multiclass_config.yaml"
    )
    parser.add_argument(
        "--ds", help="Dataset to select", default=""
    )
    parser.add_argument(
        "--dev", action='store_true', help="fast dev run"
    )
    parser.add_argument(
        "--stage",  help="fast dev run",default='Train'
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)
    if  config.classify:
        config.watch_metric = "val/F1Score_Average"
    else:
        config.watch_metric = "val/pcc"
    if config.dev:
        config.ds = 'test'
    return config

if __name__ == "__main__":
    config = init_config()
    rank_zero_info(config)
    # print(config)
    # 固定训练过程
    # FIXME: here should not be replicate
    seed_everything(config.replicate, workers=True)
    # seed_everything(44, workers=True)
    if config.model_architecture == "DrugTargetCoembedding":
        model = DrugTargetCoembeddingLightning(
            latent_dim= config.latent_dimension,
            classify=config.classify
        )
        dm = BaselineDataModule(config)
    if config.model_architecture == "BacteriaMorganAttention":
        if config.stage == 'Train':
            model = BacteriaMorganAttention(
                drug_dim=config.drug_shape,
                latent_dim=config.latent_dimension,
                classify=config.classify,
                num_classes=config.num_classes,
                loss_type=config.loss_type,
                Ensemble_Learn=config.Ensemble_Learn,
            )
            dm = BacteriaDataModule(config)
        else:
            model = BacteriaMorganAttention.load_from_checkpoint(checkpoint_path=config.checkpoint_path,
                                                                 drug_dim=config.drug_shape,
                                                                 latent_dim=config.latent_dimension,
                                                                 classify=config.classify,
                                                                 num_classes=config.num_classes,
                                                                 loss_type=config.loss_type,
                                                                 Ensemble_Learn=config.Ensemble_Learn,
                                                                 )
            dm = BacteriaPredictDataModule(config)
        metrics_callback = MetricsLossCallback(num_classes=config.num_classes, classify=config.classify)
    if config.model_architecture == "MorganAttention":
        model = MorganAttention(
            drug_dim=config.drug_shape,
            latent_dim=config.latent_dimension,
            classify=config.classify,
            num_classes=config.num_classes,
            loss_type=config.loss_type,
            Ensemble_Learn=config.Ensemble_Learn,
        )
        dm = BaselineDataModule(config)
        metrics_callback = MetricsLossCallback(num_classes=config.num_classes, classify=config.classify)
    if config.model_architecture == "MorganChembertAttention":
        if config.stage == 'Train':
            model = MorganChembertAttention(
                latent_dim=config.latent_dimension,
                classify=config.classify,
                num_classes=config.num_classes,
                loss_type=config.loss_type,
                Ensemble_Learn=config.Ensemble_Learn,
            )
            dm = MorganChembertDataModule(config)
        else:
            model = MorganChembertAttention.load_from_checkpoint(checkpoint_path=config.checkpoint_path,
                                                                 latent_dim=config.latent_dimension,
                                                                 classify=config.classify,
                                                                 num_classes=config.num_classes,
                                                                 loss_type=config.loss_type,
                                                                 Ensemble_Learn=config.Ensemble_Learn,
                                                                 )
            dm = MorganChembertPredictDataModule(config)
        metrics_callback = MetricsLossCallback(num_classes=config.num_classes, classify=config.classify)
    if config.model_architecture == "DrugTargetAttention":
        model = DrugTargetAttention(
            latent_dim=config.latent_dimension,
            classify=config.classify
        )
        if config.finetune_chembert:
            dm = FineTuneChemBertDataModule(config)
        else:
            dm = PreEncodedDataModule(config)
    strategy = strategies.DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(strategy=strategy, accelerator="gpu", devices='auto', fast_dev_run=config.dev,callbacks=[metrics_callback])
    if config.stage == 'Train':
        trainer.fit(model, datamodule=dm)
    elif config.stage == 'Test':
        trainer.validate(model, datamodule=dm)
    else:
        trainer.predict(model=model, dataloaders=dm)