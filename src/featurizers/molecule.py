import torch
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from pathlib import Path
from .base import Featurizer
from ..utils import get_logger, canonicalize



logg = get_logger()

from transformers import AutoTokenizer, AutoModel

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