import numpy as np
import torch
from pathlib import Path
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with Path("input.txt").open("r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab = ''.join(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda e: "".join([itos[i] for i in e])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# batch_size = 32
# block_size = 8
batch_size = 4
block_size = 8
# n_embd = 32
n_embd = 12
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
train_data[:block_size+1]


class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

        
class MultiHeadAttention(nn.Module):
    """
    A head is a communication channel. We can create mulitple 
    independent channels of communication that each collect different
    types of information
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

        
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        print(f"Head n_embd {n_embd}")
        print(f"Head head_size {head_size}")
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
            
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) x (B, 16, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        
        # Here is what I will communicate to you (between different heads. Gives head unique values).
        v = self.value(x) # (B, T, 16)
        out = wei @ v # (B, T, T) x (B, T, C) = (B, T, C), for dimension B is preserved

        return out

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # Init at random a learnable embedding matrix
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd)
        
        # Each position from 0 to block_size - 1 will
        # get its own positional embedding added on as an offset[
        self.position_embdding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        tok_emb = self.token_embdding_table(idx) # (B, T, C)
        pos_emb = self.position_embdding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # pos_emb is broadcasted across each b in B for tok_emb
        x = self.sa_heads(x)
        x = self.ffwd(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            # Get the prediction for the next token given the previous token(s)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            
            # Run the logits (value of embedding vector weights) through
            # softmax to get probability distribution
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution to get an index of a vector (a character)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Concatenate the idx to the current idx to create the next
            # idx vector to pass into `self` (forward pass function)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
        

@torch.no_grad() # we don't intend to do backprop so don't store all intermediate variables
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def main():
    xb, yb = get_batch("train")
    
    model = BigramLanguageModel()
    m = model.to(device)
    
    logits, loss = m(xb, yb)
    
    in_ = torch.zeros((1, 1), dtype=torch.long)
    print(in_)
    out = m.generate(in_, max_new_tokens=100)
    print(decode(out[0].tolist()))
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    
    batch_size = 32
    for iter_ in range(max_iters):
        if iter_ % eval_iters == 0:
            losses = estimate_loss(m)
            print(f"step {iter_} train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # print(loss.item())
    print(loss.item())
    
    in_ = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(in_)
    out = m.generate(in_, max_new_tokens=100)
    # print(out)
    print(decode(out[0].tolist()))

if __name__ == '__main__':
    main()