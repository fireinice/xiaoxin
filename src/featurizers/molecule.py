import pickle
import math
import torch
import pysmiles
import deepchem as dc
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from pathlib import Path
from .base import Featurizer
from ..utils import get_logger, canonicalize

from mol2vec.features import (
    mol2alt_sentence,
    mol2sentence,
    MolSentence,
    sentences2vec,
)
from gensim.models import word2vec
from torch.nn import ModuleList
from torch.nn.functional import one_hot

logg = get_logger()

MODEL_CACHE_DIR = Path(
    "/afs/csail.mit.edu/u/s/samsl/Work/Adapting_PLM_DTI/models"
)

from transformers import AutoTokenizer, AutoModel



class Mol2VecFeaturizer(Featurizer):
    def __init__(self, radius: int = 1, save_dir: Path = Path().absolute()):
        super().__init__("Mol2Vec", 300)

        self._radius = radius
        self._model = word2vec.Word2Vec.load(
            f"{MODEL_CACHE_DIR}/mol2vec_saved/model_300dim.pkl"
        )

    def _transform(self, smile: str) -> torch.Tensor:

        molecule = Chem.MolFromSmiles(smile)
        try:
            sentence = MolSentence(mol2alt_sentence(molecule, self._radius))
            wide_vector = sentences2vec(sentence, self._model, unseen="UNK")
            feats = wide_vector.mean(axis=0)
        except Exception:
            feats = np.zeros(self.shape)

        feats = torch.from_numpy(feats).squeeze().float()
        return feats


class MorganFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 2048,
        radius: int = 2,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("Morgan", shape, save_dir)

        self._radius = radius

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, self._radius, nBits=self.shape
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            logg.error(
                f"rdkit not found this smiles for morgan: {smile} convert to all 0 features"
            )
            logg.error(e)
            features = np.zeros((self.shape,))
        return features

    def _transform(self, smile: str) -> torch.Tensor:
        # feats = torch.from_numpy(self._featurizer(smile)).squeeze().float()
        feats = (
            torch.from_numpy(self.smiles_to_morgan(smile)).squeeze().float()
        )
        if feats.shape[0] != self.shape:
            logg.warning("Failed to featurize: appending zero vector")
            feats = torch.zeros(self.shape)
        return feats


class MolEFeaturizer(object):
    def __init__(self, path_to_model, gpu=0):
        self.path_to_model = path_to_model
        self.gpu = gpu
        with open(path_to_model + "/hparams.pkl", "rb") as f:
            hparams = pickle.load(f)
        self.mole = GNN(
            hparams["gnn"],
            hparams["layer"],
            hparams["feature_len"],
            hparams["dim"],
        )
        self.dim = hparams["dim"]
        if torch.cuda.is_available() and gpu is not None:
            self.mole.load_state_dict(torch.load(path_to_model + "/model.pt"))
            self.mole = self.mole.cuda(gpu)
        else:
            self.mole.load_state_dict(
                torch.load(
                    path_to_model + "/model.pt",
                    map_location=torch.device("cpu"),
                )
            )

    def transform(self, smiles_list, batch_size=None, data=None):
        if data is None:
            data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
        dataloader = GraphDataLoader(
            data,
            batch_size=batch_size
            if batch_size is not None
            else len(smiles_list),
        )
        all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                graph_embeddings = self.mole(graphs)
                res.append(graph_embeddings)
            res = torch.cat(res, dim=0).cpu().numpy()
        all_embeddings[data.parsed, :] = res
        flags[data.parsed] = True
        return all_embeddings, flags


class MolRFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 1024,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("MolR", shape, save_dir)

        self.path_to_model = f"{MODEL_CACHE_DIR}/molr_saved/gcn_1024"
        self._molE_featurizer = MolEFeaturizer(
            path_to_model=self.path_to_model
        )

    def _transform(self, smile: str) -> torch.Tensor:
        smile = canonicalize(smile)
        try:
            embeddings, _ = self._molE_featurizer.transform([smile])
        except NotImplementedError:
            embeddings = np.zeros(self.shape)
        tens = torch.from_numpy(embeddings).squeeze().float()
        return tens

class ChemBertaTokenFeaturizer(Featurizer): #只对Smiles格式化 不转化特征 转化特征在模型中处理的类
    def __init__(
            self,
            shape: int = 384,
            radius: int = 2,
            save_dir: Path = Path().absolute(),
    ):
        super().__init__("ChemBERTa", shape, save_dir)
        self._max_len = 512
        self._chemberta_tokenizer = AutoTokenizer.from_pretrained('./models/chemberta')

    def _tokenizer(self, seqs: list):

        encoded_inputs = self._chemberta_tokenizer(
            seqs,
            padding='longest',
            truncation=True,
            add_special_tokens=False,
            max_length=self._max_len,
            return_tensors='pt'
        )

        return encoded_inputs

class ChemBertaFeaturizer(Featurizer): #将Smiles转化成特征并且缓存到本地的类
    def __init__(
            self,
            shape: int = 384,
            radius: int = 2,
            save_dir: Path = Path().absolute(),
            per_tok: bool=False
    ):
        super().__init__("ChemBERTa", shape, save_dir)
        self.per_tok = per_tok
        self.tokenizer = AutoTokenizer.from_pretrained('./models/chemberta')
        self.model = AutoModel.from_pretrained('./models/chemberta')
        self._register_cuda("model", self.model)
        self.model.eval()
        self.total_error = 0

    def smiles_to_chemberta(self, smile: str) -> torch.Tensor:
        orig_smile = smile
        smile = canonicalize(smile)
        inputs = self.tokenizer(smile, add_special_tokens=False,truncation=True,return_tensors="pt")
        if self.on_cuda:
            inputs.to(self._device)
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.squeeze()
            except RuntimeError:
                self.total_error += 1
                embedding = torch.rand(16, self.shape)
                print(f"total error:{self.total_error}, smile: {orig_smile}")
        return embedding

    def _transform(self, smile: str) -> torch.Tensor:
        feats = self.smiles_to_chemberta(smile)
        if not self.per_tok : feats = feats.mean(0)
        return feats
