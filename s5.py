import torch
from torch import nn

"""
Self-attention involves the trainable weight matrics Wq, Wk, and Wv.
These matrices transform input data into queries, keys, and values, respectively,
which are crucial components of the attention mechanism.
As the model is exposed to more data during training, it adjusts these trainable weights.
"""

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, dkv_bias=False):
        super().__init__()
        # nn.Linear effectively perform matrix multiplication when the bias units are disabled
        # Additionally, nn.Linear has an optimized weight initialization scheme,
        # contributing to more stable and effective model training.
        self.W_quest = nn.Linear(d_in, d_out, bias=dkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=dkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=dkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_quest(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
