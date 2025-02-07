import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

seed = 1337
torch.manual_seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Seed {seed}")
print(f"Device {device}")

@dataclass
class Config:
    max_iters : int = 10_000
    eval_iters : int = 200
    batch_size : int = 4
    n_emb : int = 32
    cw_size : int = 8
    n_embd : int = 32

    vocab : str = None
    vocab_size : int = None
    train_split : float = 0.9

class Data:

    def __init__(self, text, config):
        vocab = sorted(list(set(text)))
        self.itos = { ch: i for i, ch in enumerate(vocab)}
        self.stoi = { i: ch for i, ch in enumerate(vocab)}
        config.vocab = vocab
        config.vocab_size = len(vocab)

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(config.train_split * len(data))
        self.train_data = data[:n]
        self.test_data = data[n:]

    def encode(self, x):
        return [self.itos[s] for s in x]

    def decode(self, x):
        return [self.stoi[s] for s in x]
        
    def get_batch(self, split, config):
        data = self.train_data if split == 'train' else self.test_data
        t = config.cw_size
        b = config.batch_size
        indexes = torch.randint(len(data) - t, (b,))
        x = torch.stack([data[ix:ix + t] for ix in indexes]).to(device)
        y = torch.stack([data[ix + 1:ix + t + 1] for ix in indexes]).to(device)

        return x, y
    
    @torch.no_grad
    def estimate_loss(self, model, config):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            for i in range(config.eval_iters):
                X, Y = self.get_batch(split, config)
                logits, loss = model(X, Y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()

        return out


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        tok_emb = self.tok_emb(idx) # (b, t, d)
        logits = self.lm_head(tok_emb) # (b, t, v), where v = vocab_size

        if targets is None:
            loss = None
        else:
            b, t, v = logits.shape
            logits = logits.view(b*t, v)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.cw_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx 

def train_test_model(config : Config):
    with Path("../../input.txt").open("r", encoding="utf-8") as f:
        text = f.read()

    config = Config()
    data = Data(text, config)

    model = LanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    with tqdm(range(config.max_iters), desc="Training", leave=False) as pbar:
        for iter_ in pbar:
            # Estimate train/val loss
            if iter_ % config.eval_iters == 0:
                out = data.estimate_loss(model, config)

            # Forward pass
            xb, yb = data.get_batch('train', config)
            optimizer.zero_grad()
            logits, loss = model(xb, yb)

            # Tqdm write progress bar
            pbar.set_postfix(Train_Loss=out['train'], Val_Loss=out['val'])
            
            # Update weights
            loss.backward()
            optimizer.step()

    # Test generate loop
    model.eval()
    in_tensor = torch.zeros((1, 1), dtype=torch.long)
    out_tensor = model.generate(in_tensor)
    # print(out_tensor)
    print("".join(data.decode(out_tensor[0].tolist())))



def test_architecture(config : Config):
    pass

def main():
    config = Config
    test_architecture(config)
    train_test_model(config)

if __name__ == '__main__':
    main()
