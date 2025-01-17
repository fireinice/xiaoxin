from pathlib import Path
import torch
from . import Featurizer
from ..utils import get_logger
from transformers import AutoTokenizer, AutoModel, pipeline, T5Tokenizer, T5EncoderModel, AutoModelForCausalLM

logg = get_logger()
FOLDSEEK_MISSING_IDX = 0

class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), name="ProtBert", per_tok=False):
        super().__init__(name, 1024, save_dir)

        self._max_len = 1024
        self.per_tok = per_tok

        self._tokenizer = AutoTokenizer.from_pretrained("./models/probert", do_lower_case=False)
        self._model = AutoModel.from_pretrained("./models/probert")
        self._featurizer = pipeline("feature-extraction", model=self._model, tokenizer=self._tokenizer)

        self._register_cuda("model", self._model)
        self._register_cuda("featurizer", self._featurizer, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline

        d = device.index if device.type != "cpu" else -1

        pipe = pipeline("feature-extraction", model=self._model, tokenizer=self._tokenizer, device=d)
        self._featurizer = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        embedding = torch.tensor(self._cuda_registry["featurizer"][0](self._space_sequence(seq)))
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        return feats if self.per_tok else feats.mean(0)

class ProtT5Featurizer(ProtBertFeaturizer):
    def __init__(self, save_dir: Path = Path().absolute(), name="ProtT5", per_tok=False):
        super().__init__(save_dir, name, per_tok)

        self._model = T5EncoderModel.from_pretrained("./models/prot5")
        self._tokenizer = T5Tokenizer.from_pretrained("./models/prot5", do_lower_case=False)
        self._featurizer = pipeline("feature-extraction", model=self._model, tokenizer=self._tokenizer)
        self._register_cuda("model", self._model)
        self._register_cuda("featurizer", self._featurizer, self._feat_to_device)

class Esm2Featurizer(ProtBertFeaturizer):
    def __init__(self, save_dir: Path = Path().absolute(), name="Esm2", per_tok=False):
        super().__init__(save_dir, name, per_tok)

        self._tokenizer = AutoTokenizer.from_pretrained("./models/esm2", do_lower_case=False)
        self._model = AutoModel.from_pretrained("./models/esm2")
        self._featurizer = pipeline("feature-extraction", model=self._model, tokenizer=self._tokenizer)
        self._register_cuda("model", self._model)
        self._register_cuda("featurizer", self._featurizer, self._feat_to_device)

class Protgpt2Featurizer(ProtBertFeaturizer):
    def __init__(self, save_dir: Path = Path().absolute(), name="Protgpt2", per_tok=False):
        super().__init__(save_dir, name, per_tok)

        self._tokenizer = AutoTokenizer.from_pretrained("./models/protgpt2", do_lower_case=False)
        self._model = AutoModelForCausalLM.from_pretrained("./models/protgpt2")
        self._featurizer = pipeline("feature-extraction", model=self._model, tokenizer=self._tokenizer)
        self._register_cuda("model", self._model)
        self._register_cuda("featurizer", self._featurizer, self._feat_to_device)


    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        input_ids = self._tokenizer.encode(seq, return_tensors="pt").to(self.device)
        outputs = self._model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embedding = hidden_states[-1]
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        return feats if self.per_tok else feats.mean(0)

