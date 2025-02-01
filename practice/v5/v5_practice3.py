import torch
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch import nn
from dataclasses import dataclass

seed = 1337
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE {device}")
print(f"SEED {seed}")

@dataclass
class Config:
    cw_size          : int      = 8
    train_test_split : float    = 0.9
    batch_size       : int      = 4
    n_embd           : int      = 32
    n_heads          : float    = 2
    n_layers         : int      = 2
    max_iters        : int      = 5000
    eval_iters        : int     = 200

    # Dropout
    p_dropout        : float    = 0.1
    
    # Layer Norm hyperparams
    eps              : float   = 1e-5
    momentum         : float   = 0.01
    
class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We initialize the scale (gamma) and shift/bias (beta)
        # parameters to ones and zeros respectively to 
        # ensure the initial operation is the identity (standard normal)
        # That way the network can decide what it wants to do
        # with the values from there
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd))
    
    def forward(self, x):
        x_mean = x.mean(-1, keepdims=True)
        x_var = x.var(-1, keepdims=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.config.eps)
        out = x_hat * self.gamma + self.beta
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_heads
        self.att_heads = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.att_heads], dim=-1)
        out = self.proj(x)
        return out

class AttentionHead(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        
        tril = torch.tril(torch.ones(config.n_embd, config.n_embd))
        self.register_buffer('tril', tril)
        self.dropout = nn.Dropout(p=config.p_dropout)

    def forward(self, x):
        B, T, d = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        att = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(p=config.p_dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.mha(x)
        x = self.ln2(x)
        out = x + self.ffwd(x)
        return out

class MiniGpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.cw_size, config.n_embd)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        # In VIT we do layer-normalization here before lm head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.transformer_blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            losses = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            losses = F.cross_entropy(logits, targets)
            
        return logits, losses

    def generate(self, idx, max_new_tokens=1000):
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            # B, T = idx.shape
            idx_cond = idx[:, -self.config.cw_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

class Data:
    def encode(self, x):
        return [self.stoi[s] for s in x]
        
    def decode(self, x):
        return [self.itos[s] for s in x]
        
    def __init__(self, config, text):
        self.config = config
        vocab = sorted(list(set(text)))
        config.vocab = vocab
        config.vocab_size = len(vocab)
        
        self.stoi = { ch: i for i, ch in enumerate(vocab) }
        self.itos = { i: ch for i, ch in enumerate(vocab) }

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(config.train_test_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.cw_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.cw_size] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+self.config.cw_size+1] for i in ix]).to(device)

        return x, y

    @torch.no_grad()
    def estimate_loss(self, model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            Losses = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                logits, losses = model(X, Y)
                Losses[i] = losses.item()
            out[split] = Losses.mean().item() 
        return out
        
def test_architecture(config : Config):
    # -----------
    # LayerNorm
    # -----------
    module = LayerNorm(config)
    B, T, d = (16, 8, 32)
    in_tensor = torch.ones(B, T, d)
    print(f"in_tensor.shape {in_tensor.shape}")
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    print()
    
    # -----------
    # AttentionHead
    # -----------
    B, T, d = (16, 8, 32)
    head_size = d // 4
    module = AttentionHead(config, head_size)
    in_tensor = torch.ones(B, T, d)
    print(f"in_tensor.shape {in_tensor.shape}")
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    print()
    
    # -----------
    # MultiHeadAttention
    # -----------
    B, T, d = (16, 8, 32)
    module = MultiHeadAttention(config)
    in_tensor = torch.ones(B, T, d)
    print(f"in_tensor.shape {in_tensor.shape}")
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    print()
    
    # -----------
    # TransformerBlock
    # -----------
    B, T, d = (16, 8, 32)
    module = TransformerBlock(config)
    in_tensor = torch.ones(B, T, d)
    print(f"in_tensor.shape {in_tensor.shape}")
    out_tensor = module(in_tensor)
    print(f"out_tensor.shape {out_tensor.shape}")
    print()
    
def main():
    config = Config()
    
    with Path("../../input.txt").open("r", encoding="utf-8") as f:
        text = f.read()

    data = Data(config, text)
    test_architecture(config)


    model = MiniGpt(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pbar = tqdm(range(config.max_iters))
    for iter_ in pbar:
        # Every eval_iters iterations, evaluate and log losses
        if iter_ % config.eval_iters == 0:
            out = data.estimate_loss(model)
            # Update the progress bar's postfix with the current train and validation loss
            pbar.set_postfix({
                'Train Loss': f"{out['train']:.4f}",
                'Val Loss': f"{out['val']:.4f}"
            })
            # Print the losses below the progress bar to build up a history log
            tqdm.write(f"Iter {iter_:5}: Train Loss: {out['train']:.4f} | Val Loss: {out['val']:.4f}")
        
        # Standard training step
        x, y = data.get_batch('train')
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

    
    out = model.generate(torch.zeros((1, 1), dtype=torch.long))
    print("".join(data.decode(out[0].tolist())))
    
if __name__ == '__main__':
    main()






















        