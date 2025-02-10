import torch
import torch.nn.functional as F
        
from torch.utils.data import IterableDataset, DataLoader

from torch import nn
from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, x):
        x = F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sa_heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        x = self.proj(x)
        if self.config.use_dropout:
            x = self.dropout(x)
        
        return x

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_size = config.n_embd // config.n_heads
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)

        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer('tril', tril)
        
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        if self.config.use_dropout:
            wei = self.dropout(wei)
        out = wei @ v
        
        return out

# def LanguageModel
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ma_head = MultiHeadAttention(config)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout),
        )
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)
            

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.ma_head(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)

        return x

class DecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.cw_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        idx.to(device)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.lm_head(x)
        if targets is None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
        return logits, losses

    def generate(self, idx, config, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.cw_size:]
            logits, _ = self(idx_cond) # logits.shape => (B, T, vocab_size)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
            

@dataclass
class Config:
    n_embd : int = 32
    n_heads : int = 4
    n_blocks : int = 3
    cw_size : int = 16
    batch_size : int = 128

    max_iters : int = 3000
    eval_iters : int = 200

    vocab_size : int = None

    use_dropout : bool = False
    p_dropout : float = 0.2

    bias : bool = True
    
class Data:
    def encode(self, x):
        return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def __init__(self, text, config):
        vocab = sorted(list(set(text)))
        config.vocab_size = len(vocab)
        self.stoi = { ch: i for i, ch in enumerate(vocab)}
        self.itos = { i: ch for i, ch in enumerate(vocab)}
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, config):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - config.cw_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.cw_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+config.cw_size+1] for i in ix]).to(device)
        
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model, config):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            Losses = torch.zeros(config.eval_iters)
            for i in range(config.eval_iters):
                X, Y = self.get_batch(split, config)
                logits, losses = model(X, Y)
                Losses[i] = losses.item()
            out[split] = Losses.mean()
        model.train()
        return out

def main():
    config = Config()
    with Path("../../input.txt").open("r", encoding="utf-8") as f:
        text = f.read()
    data = Data(text, config)
    model = DecoderLM(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for step in range(config.max_iters):
        if step % config.eval_iters == 0:
            out = data.estimate_loss(model, config)
            print(f"train loss {out['train']:.4f} val loss {out['val']:.4f}")
        xb, yb = data.get_batch('train', config)
        logits, losses = model(xb, yb)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    out = model.generate(torch.zeros((1, 1), dtype=torch.long), config)
    print("".join(data.decode(out[0].tolist())))
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
