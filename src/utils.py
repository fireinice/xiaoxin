import torch
import sys
import h5py
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from functools import partial

import logging as lg
import typing as T
import multiprocessing as mp

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logLevels = {0: lg.ERROR, 1: lg.WARNING, 2: lg.INFO, 3: lg.DEBUG}
LOGGER_NAME = "DTI"


def get_logger():
    return lg.getLogger(LOGGER_NAME)


logg = get_logger()


def config_logger(
    file: T.Union[Path, None],
    fmt: str,
    level: bool = 2,
    use_stdout: bool = True,
):
    """
    Create and configure the logger

    :param file: Can be a Path or None -- if a Path, log messages will be written to the file at Path
    :type file: T.Union[Path, None]
    :param fmt: Formatting string for the log messages
    :type fmt: str
    :param level: Level of verbosity
    :type level: int
    :param use_stdout: Whether to also log messages to stdout
    :type use_stdout: bool
    :return:
    """

    module_logger = lg.getLogger(LOGGER_NAME)
    module_logger.setLevel(logLevels[level])
    formatter = lg.Formatter(fmt)

    if file is not None:
        fh = lg.FileHandler(file)
        fh.setFormatter(formatter)
        module_logger.addHandler(fh)

    if use_stdout:
        sh = lg.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        module_logger.addHandler(sh)

    lg.propagate = False

    return module_logger


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return smiles


def smiles2morgan(s, radius=2, nBits=2048):
    """
    Convert smiles into Morgan Fingerprint.
    :param smile: SMILES string
    :type smile: str
    :return: Morgan fingerprint
    :rtype: np.ndarray
    """
    try:
        s = canonicalize(s)
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits
        )
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except Exception as e:
        logg.error(e)
        logg.error(
            f"Failed to convert SMILES to Morgan Fingerprint: {s} convert to all 0 features"
        )
        features = np.zeros((nBits,))
    return features


def get_config(experiment_id, mol_feat, prot_feat):
    data_cfg = {
        "batch_size": 32,
        "num_workers": 0,
        "precompute": True,
        "mol_feat": mol_feat,
        "prot_feat": prot_feat,
    }
    model_cfg = {
        "latent_size": 1024,
    }
    training_cfg = {
        "n_epochs": 50,
        "every_n_val": 1,
    }
    cfg = {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "experiment_id": experiment_id,
    }

    return OmegaConf.structured(cfg)


def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p


def _hdf5_load_partial_func(k, file_path):
    """
    Helper function for load_hdf5_parallel
    """

    with h5py.File(file_path, "r") as fi:
        emb = torch.from_numpy(fi[k][:])
    return emb


def load_hdf5_parallel(file_path, keys, n_jobs=-1):
    """
    Load keys from hdf5 file into memory
    :param file_path: Path to hdf5 file
    :type file_path: str
    :param keys: List of keys to get
    :type keys: list[str]
    :return: Dictionary with keys and records in memory
    :rtype: dict
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    if (n_jobs == -1) or (n_jobs > mp.cpu_count()):
        n_jobs = mp.cpu_count()

    with mp.Pool(processes=n_jobs) as pool:
        all_embs = list(
            tqdm(
                pool.imap(
                    partial(_hdf5_load_partial_func, file_path=file_path), keys
                ),
                total=len(keys),
            )
        )

    embeddings = {k: v for k, v in zip(keys, all_embs)}
    return embeddings


def get_featurizer(featurizer_string, *args, **kwargs):
    from . import featurizers

    featurizer_string_list = featurizer_string.split(",")
    if len(featurizer_string_list) > 1:
        featurizer_list = [
            getattr(featurizers, i.strip())(*args, **kwargs) for i in featurizer_string_list
        ]
        # return featurizers.ConcatFeaturizer(featurizer_list, *args, **kwargs)
        return featurizer_list
    else:
        return getattr(featurizers, featurizer_string_list[0])(*args, **kwargs)

def get_task_dir(task_name: str):

    task_paths = {
        "biosnap": "./dataset/BIOSNAP/full_data",
        "biosnap_prot": "./dataset/BIOSNAP/unseen_protein",
        "biosnap_mol": "./dataset/BIOSNAP/unseen_drug",
        "bindingdb": "./dataset/BindingDB",
        "davis": "./dataset/DAVIS",
        "dti_dg": "./dataset/TDC",
        "dude": "./dataset/DUDe",
        "halogenase": "./dataset/EnzPred/halogenase_NaCl_binary",
        "bkace": "./dataset/EnzPred/duf_binary",
        "gt": "./dataset/EnzPred/gt_acceptors_achiral_binary",
        "esterase": "./dataset/EnzPred/esterase_binary",
        "kinase": "./dataset/EnzPred/davis_filtered",
        "phosphatase": "./dataset/EnzPred/phosphatase_chiral_binary",
        "bindingdb_v2": "./dataset/BingdingDB_v2",
        "bindingdb_mc": "./dataset/BindingDB_MC",
        "bacteria_predict": "./dataset/bacteria_predict"
    }

    return Path(task_paths[task_name.lower()]).resolve()
