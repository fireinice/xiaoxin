import yaml

config = {
    "task": "BindingDB",
    "experiment_id": "bindingdb_morgan_chembert_ankh_attention",
    "drug_featurizer": "MorganFeaturizer,ChemBertaFeaturizer",
    "drug_shape": 2048,
    "target_featurizer": "AnkhFeaturizer",
    "target_shape": 1024,
    "model_architecture": "MorganChemBertAttention",
    "latent_dimension": 1024,
    "classify": True,
    "num_classes": 5,
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 8,
    "loss_type": "OR",
    "ensemble_learn": False,
    "lr": 1e-4,
    "lr_t0": 10,
    "replicate": 0,
    "device": 0,
    "checkpoint_path": "",
    "bins": [0, 50, 200, 1000, 10000]
}

yaml_file_path = f"configs/{config['experiment_id']}.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

print(f"YAML configuration has been saved to {yaml_file_path}.")
