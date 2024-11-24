import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything, strategies
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf

from src.models.drug_target_attention import DrugTargetAttention
from src.datamodule.pre_encoded_datamodule import PreEncodedDataModule


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
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(arg_overrides)
    if config.model_architecture == "ChemBertaProteinAttention_Local":
        config.classify = False
        config.watch_metric = "val/pcc"
    else:
        config.classify = False
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
    model = DrugTargetAttention(
        latent_dim=config.latent_dimension,
        classify=config.classify,
        lr=config.lr,
        lr_t0=config.lr_t0,
    )
    dm = PreEncodedDataModule(config)
    strategy = strategies.DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(strategy=strategy, accelerator="gpu", devices='auto', fast_dev_run=config.dev)
    trainer.fit(model, datamodule=dm)
