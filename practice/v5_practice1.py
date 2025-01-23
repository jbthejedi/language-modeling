import torch
import torch.nn.functional as F

from torch import nn
from pathlib import Path

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout_p = 0.2

class Data:
    def encode(self, x):
        return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = { ch: i for i, ch in enumerate(self.vocab) }
        self.itos = { i: ch for i, ch in enumerate(self.vocab) }

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    @torch.no_grad()
    def estimate_loss(self, model, eval_iters):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            Losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                X, Y = self.get_batch(split)
                logits, losses = model(X, Y)
                Losses[i] = losses.item()
            out[split] = Losses.mean()
        model.train()
        return out
        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y

class Block(nn.Module):
    def __init__(self, n_embd, block_size, n_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_embd, block_size, n_heads)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(p=dropout_p)
        )
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa_heads(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x

class Head(nn.Module):
    def __init__(self, n_embd, block_size, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def forward(self, x):
        xmean = x.mean(-1, keepdims=True)
        xvar = x.var(-1, keepdims=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = xhat*self.gamma + self.beta
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]
        
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size, n_heads):
        super().__init__()
        self.sa_heads = nn.ModuleList([Head(n_embd, block_size, head_size=n_embd // n_heads) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.sa_heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
        
class LanguageModel(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, n_heads):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd, block_size)
        # self.sa_heads = MultiHeadAttention(n_embd, block_size, n_heads)
        self.trans_blocks = nn.Sequential(
            Block(n_embd, block_size, n_heads),
            Block(n_embd, block_size, n_heads),
            Block(n_embd, block_size, n_heads),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        # x = self.sa_heads(x)
        x = self.trans_blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            losses = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
        return logits, losses

    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx



















        