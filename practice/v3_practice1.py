from torch import nn
from torch.nn import functional as F
from pathlib import Path

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
        
        
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        # print(f"Head n_embd {n_embd}")
        # print(f"Head head_size {head_size}")
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # print(f"Head btc {B, T, C}")
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        v = self.value(x) # (B, T, C)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        # print(f"mha head_size {head_size}")
        self.heads = nn.ModuleList([Head(n_embd=n_embd, head_size=head_size, block_size=block_size) for _ in range(num_heads)])
        # Each head is (B, T, n_embd // num_heads)
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # print(f"LM n_embd {n_embd} num_heads {num_heads}")
        # self.sa_heads = MultiHeadAttention(
        #     num_heads=num_heads,
        #     head_size=n_embd // num_heads, # n_embd should be neatly divisible by num_heads
        #     block_size=block_size
        # )
        self.sa_heads = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            head_size=n_embd // num_heads, # n_embd should be neatly divisible by num_heads
            block_size=block_size
        )
        # self.sa_head = Head(head_size=n_embd, block_size=block_size)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        x = self.ffwd(x)
        logits = self.lm_head(x)
        
        if targets is None:
            losses = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
        return logits, losses

    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

def main():
    with Path('../input.txt').open('r', encoding='utf-8') as f:
        text = f.read()
    
    vocab = sorted(list(set(text)))
    stoi = { ch: i for i, ch in enumerate(vocab) }
    itos = { i: ch for i, ch in enumerate(vocab) }
    encode = lambda x: [stoi[s] for s in x]
    decode = lambda x: [itos[s] for s in x]
    
    print("".join(decode(encode("hii there"))))
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[n:]
    val_data = data[:n]
    
    batch_size = 4
    block_size = 8
    n_embd = 12
    eval_iters = 200
    max_iters = 4000
    vocab_size = len(vocab)
    
    def get_batch(split, batch_size):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
        
    @torch.no_grad()
    def estimate_loss(model, eval_iters, batch_size):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            Losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                X, Y = get_batch(split, batch_size)
                # print(X.shape, Y.shape)
                # break
                logits, losses = model(X, Y)
                Losses[i] = losses.item()
            out[split] = Losses.mean()
        model.train()
        return out
    
    
    m = LanguageModel(vocab_size, n_embd, block_size, num_heads=4)
    m = m.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for iter_ in range(max_iters):
        if iter_ % eval_iters == 0:
            out = estimate_loss(m, eval_iters, batch_size)
            print(f"train loss {out['train']} val loss {out['val']}")
        xb, yb = get_batch('train', batch_size)
        logits, losses = m(xb, yb)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f"final loss {losses.item()}")
    
    _in = torch.zeros((1, 1), dtype=torch.long)
    out = m.generate(_in, max_new_tokens=1000)
    print("".join(decode(out[0].tolist())))