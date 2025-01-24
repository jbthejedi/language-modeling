import torch
import torch.nn.functional as F

from torch import nn
from pathlib import Path
from dataclasses import dataclass

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Data:
    def encode(self, x):
        return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def __init__(self, config):
        self.vocab = sorted(list(set(config.text)))
        config.vocab_size = len(self.vocab)
        
        self.stoi = { ch: i for i, ch in enumerate(self.vocab) }
        self.itos = { i: ch for i, ch in enumerate(self.vocab) }
        
        data = torch.tensor(self.encode(config.text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[n:]
        self.val_data = data[:n]

    def get_batch(self, split, config):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - config.cw_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.cw_size] for i in ix])
        y = torch.stack([data[i+1:i+config.cw_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad
    def estimate_loss(self, model, config):
        out = {}
        for split in ['train', 'val']:
            Losses = torch.zeros(config.eval_iters)
            for i in range(config.eval_iters):
                X, Y = self.get_batch(split, config)
                logits, losses = model(X, Y)
                Losses[i] = losses.item()
            out[split] = Losses.mean()
        return out

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        
        tril = torch.tril(torch.ones(config.cw_size, config.cw_size))
        self.register_buffer('tril', tril)

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.sa_heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_heads)])

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_heads = MultiHeadAttention(config)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sa_heads(x)
        x = self.ffwd(x)
        return x

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.cw_size, config.n_embd)
        self.sa_blocks = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_blocks(x)
        logits = self.lm_head(x)
        
        if targets == None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
            
        return logits, losses
        
    def generate(self, idx, max_new_tokens=1000):
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.cw_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
        

@dataclass
class Config:
    max_iters: int = 5000
    eval_iters: int = 200
    text: str = ""
    
    cw_size: int = 8
    batch_size: int = 4
    n_embd: int = 32
    n_heads: int = 4
    vocab_size: int = None
    
def main():
    config = Config()
    # with Path("/root/language-modeling/practice/v5/input.txt").open("r", encoding="utf-8") as f:
    with Path("../../input.txt").open("r", encoding="utf-8") as f:
        config.text = f.read()
    data = Data(config)

    # Model
    m = LanguageModel(config)
    m = m.to(device)

    # Train
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for iter_ in range(config.max_iters):
        if iter_ % config.eval_iters == 0:
            out = data.estimate_loss(m, config)
            print(f"train loss {out['train']} val loss {out['val']}")
        xb, yb = data.get_batch('train', config)
        logits, losses = m(xb, yb)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    # Generate
    m.eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long).to(device))
    print("".join(data.decode(out[0].tolist())))

if __name__ == 'main':
    main()
























        