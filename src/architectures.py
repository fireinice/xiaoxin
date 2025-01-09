import torch
from typing import Optional
from .utils import get_logger
from torch import nn

logg = get_logger()

class MLP(nn.Module):

    def __init__(self, inputdim: int, hiddendim:int, outdim:int) -> None:
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(inputdim, hiddendim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hiddendim, outdim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        return self.mlp(inputs)

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, max_position_embeddings, hidden_size, type_size):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
       
        input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, 0 : seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664

       
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

