from torch import nn
from torch.nn import functional as F
from pathlib import Path
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Batch-norm normalizes mn, std for each feature

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            n_embd=n_embd,
            head_size=n_embd // n_head,
            block_size=block_size,
        )
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
        
        
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # T can be < block_size so select `:T`
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        
        # The `value` weights are what I will communicate to you
        # (between different heads. Gives head unique values).
        v = self.value(x) # (B, T, C)
        out = wei @ v
        return out
        

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                block_size=block_size,
            ) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return x

        
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.block(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
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

class Data:
    def __init__(self, text, d, block_size):
        self.block_size = block_size
        
        self.vocab = sorted(list(set(text)))
        self.stoi = { ch: i for i, ch in enumerate(self.vocab) }
        self.itos = { i: ch for i, ch in enumerate(self.vocab) }
        
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        
        n = int(d * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        
    def encode(self, x):
         return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        # print(f"ix.shape {ix.shape}")
        # print(ix)
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            Losses = torch.zeros(self.eval_iters)
            for i in range(self.eval_iters):
                X, Y = self.get_batch(split)
                # print(X.shape, Y.shape)
                logits, losses = model(X, Y)
                Losses[i] = losses
            out[split] = Losses.mean()
        model.train()
        return out

        
def main():
    with Path("../input.txt").open("r", encoding='utf-8') as f:
        text = f.read()

    data = Data(text, d=0.9, block_size=8)
    data.max_iters = 4000
    data.eval_iters = 200
    data.batch_size = 32
    
    m = LanguageModel(
        vocab_size=len(data.vocab),
        n_embd=32,
        block_size=data.block_size,
        num_heads=4,
    )
    m = m.to(DEVICE)

    # Train
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m.train()
    for iter_ in range(data.max_iters):
        if iter_ % data.eval_iters == 0:
            out = data.estimate_loss(m)
            print(f"train loss {out['train']} val loss {out['val']}")
        xb, yb = data.get_batch('train')
        logits, losses = m(xb, yb)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"final loss {losses.item()}")
    
    _in = torch.zeros((1, 1), dtype=torch.long)
    out = m.generate(_in, max_new_tokens=1000)
    print("".join(data.decode(out[0].tolist())))



            



    


            