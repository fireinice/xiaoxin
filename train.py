import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from src.datamodule.bacteria_predict_datamodule import BacteriaPredictDataModule
from src.datamodule.bacteria_datamodule import BacteriaDataModule
from src.datamodule.baseline_datamodule import BaselineDataModule
from src.datamodule.morgan_chembert_predict_datamodule import MorganChembertPredictDataModule
from src.models.bacteria_morgan_model import BacteriaMorganAttention
from src.callback.metrics_callback import MetricsCallback
from src.models.lightning_model import DrugTargetCoembeddingLightning
from src.models.morgan_chembert_model import MorganChembertAttention
from src.models.morgan_model import MorganAttention

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything, strategies
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf

# from src.models.drug_target_attention import DrugTargetAttention
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
        "--config", help="YAML config file", default="configs/chembert_muilt.yaml"
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
    if config.ds == 'test':
        config.batch_size = 4
        config.num_workers = 2
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
                ensemble_learn=config.ensemble_learn,
            )
            dm = BacteriaDataModule(config)
        else:
            model = BacteriaMorganAttention.load_from_checkpoint(checkpoint_path=config.checkpoint_path,
                                                                 drug_dim=config.drug_shape,
                                                                 latent_dim=config.latent_dimension,
                                                                 classify=config.classify,
                                                                 num_classes=config.num_classes,
                                                                 loss_type=config.loss_type,
                                                                 ensemble_learn=config.ensemble_learn,
                                                                 )
            dm = BacteriaPredictDataModule(config)
    if config.model_architecture == "MorganAttention":
        model = MorganAttention(
            drug_dim=config.drug_shape,
            latent_dim=config.latent_dimension,
            classify=config.classify,
            num_classes=config.num_classes,
            loss_type=config.loss_type,
            ensemble_learn=config.ensemble_learn,
        )
        dm = BaselineDataModule(config)
    if config.model_architecture == "MorganChembertAttention":
        if config.stage == 'Train':
            model = MorganChembertAttention(
                latent_dim=config.latent_dimension,
                classify=config.classify,
                num_classes=config.num_classes,
                loss_type=config.loss_type,
                ensemble_learn=config.ensemble_learn,
                target_dim=config.target_shape
            )
            dm = MorganChembertDataModule(config)
        else:
            model = MorganChembertAttention.load_from_checkpoint(checkpoint_path=config.checkpoint_path,
                                                                 latent_dim=config.latent_dimension,
                                                                 classify=config.classify,
                                                                 num_classes=config.num_classes,
                                                                 loss_type=config.loss_type,
                                                                 Ensemble_Learn=config.ensemble_learn,
                                                                 )
            dm = MorganChembertPredictDataModule(config)
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
    metrics_callback = MetricsCallback(num_classes=config.num_classes, classify=config.classify)
    fn = f"{config.experiment_id}-epoch={{epoch}}-metric={{{config.watch_metric}:.2f}}"
    metric_save_callback = ModelCheckpoint(
        monitor=config.watch_metric,
        dirpath='ckpts',
        filename=fn,
        save_top_k=3,
        mode='max',
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False
    )
    trainer = Trainer(strategy=strategy, accelerator="gpu", devices=[0, 1], fast_dev_run=config.dev,callbacks=[metrics_callback,metric_save_callback])
    if config.stage == 'Train':
        trainer.fit(model, datamodule=dm)
    elif config.stage == 'Test':
        trainer.validate(model, datamodule=dm)
    else:
        trainer.predict(model=model, dataloaders=dm)
