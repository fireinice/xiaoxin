from transformers import AutoTokenizer, AutoModel
from src.utils import get_logger, canonicalize
import torch

tokenizer = AutoTokenizer.from_pretrained('./models/chemberta')
model = AutoModel.from_pretrained('./models/chemberta')
model.eval()

chem = "CSCC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](C)NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CO)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CO)NC(=O)[C@@H](C)NC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)CNC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)CO)[C@@H](C)O)C(C)C)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H]([C@@H](C)O)C(N)=O"

smile = canonicalize(chem)

if smile is not None:
    inputs = tokenizer(smile, padding="max_length", max_length=512, return_tensors="pt")

    print(inputs['input_ids'].shape)

    with torch.no_grad():

        output = model(**inputs)
