import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class PracticeConfig:
    max_iters : int = 5000
    eval_iters : int = 200
    batch_size : int = 4
    n_head : int = 4
    n_emb : int = 32
    cw_size : int = 8
    n_embd : int = 32
    p_dropout : float = 0.2

    vocab : str = None
    vocab_size : int = None
    train_split : float = 0.9

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)

        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer('tril', tril)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        b, t, d = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = q @ k.transpose(1, 2) * (self.head_size**-0.5)
        att = att.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v

        return out

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = MyLayerNorm(dim=config.n_embd)
        self.ln2 = MyLayerNorm(dim=config.n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln1(x)
        x = x + self.ffwd(x)
        return x

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdims=True)
        x_var = x.var(-1, keepdims=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        x_hat = x_hat * self.gamma + self.beta

        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.attention_heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        x = self.proj(x)
        return x

def test_architecture():
    config = PracticeConfig()
    head_size = config.n_embd // config.n_head
    module = Head(config, head_size)
    in_tensor = torch.zeros(4, 6, 32)
    out_tensor = module(in_tensor)
    assert out_tensor.shape == (4, 6, 8), f"""
    Failed. Expected (4, 6, 8) but got {out_tensor.shape}
    """

    config = Config()
    module = MultiHeadAttention(config)
    in_tensor = torch.zeros(4, 6, 32)
    out_tensor = module(in_tensor)
    assert out_tensor.shape == (4, 6, 32), f"""
    Failed. Expected (4, 6, 32) but got {out_tensor.shape}
    """

if __name__ == '__main__':
    test_architecture()
