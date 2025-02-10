import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_modules as tm

from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337
torch.manual_seed(seed)
print(f"Device {device}")
print(f"Seed {seed}")

@dataclass
class Config:
    cw_size : int = 8
    batch_size : int = 4
    train_split : float = 0.9
    n_embd : int = 32
    max_iters : int = 5000
    eval_iters : int = 500
    p_dropout : float = 0.1
    n_heads : int = 4
    n_blocks : int = 2

class Data:
    def __init__(self, text, config):
        tokens = word_tokenize(text)
        # vocab = list(set(text))
        vocab = sorted(list(set(tokens)))
        config.vocab = vocab
        config.vocab_size = len(vocab)
        self.stoi = { ch: i for i, ch in enumerate(vocab) }
        self.itos = { i: ch for i, ch in enumerate(vocab) }

        # data = torch.tensor(self.encode(text), dtype=torch.long)
        data = torch.tensor(self.encode(tokens), dtype=torch.long)

        n = int(config.train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, tokens):
        return [self.stoi[token] for token in tokens]
    
    def decode(self, tokens):
        return [self.itos[token] for token in tokens]

    def get_batch(self, split, config):
        data = self.train_data if split == 'train' else self.val_data
        idxs = torch.randint(len(data) - config.cw_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.cw_size] for i in idxs]).to(device)
        y = torch.stack([data[i+1:i+config.cw_size+1] for i in idxs]).to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model, config):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_iters)
            for i in range(config.eval_iters):
                X, Y = self.get_batch(split, config)
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
        

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding_table = nn.Embedding(config.cw_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[tm.TransformerBlock(config) for _ in range(config.n_blocks)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, inputs, targets=None):
        b, t = inputs.shape
        tok_emb = self.token_embedding_table(inputs)
        pos_emb = self.pos_embedding_table(torch.arange(t, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        # b = batch size, t = cw size,
        # d = num dimensions of word embeddings
        if targets is not None:
            b, t, d = logits.shape

            logits = logits.view(b*t, -1)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss
    
    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.cw_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

def train_test_model(config : Config):
    # To download the data, run the below in bash or jupyter notebook
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    with Path("../../input.txt").open("r", encoding="utf-8") as f:
        text = f.read()
    data = Data(text, config)
    print(f"config.vocab_size {config.vocab_size}")

    # Train
    model = LanguageModel(config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pbar = tqdm(range(config.max_iters), position=0, leave=False)
    for iter_ in pbar:
        # Call estimate loss function
        if iter_ % config.eval_iters == 0:
            out = data.estimate_loss(model, config)
            pbar.set_postfix(Train_Loss=out['train'].item(), Val_Loss=out['val'].item())
        xb, yb = data.get_batch('train', config)
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        
    # Generate
    in_tensor = torch.zeros((1, 1), dtype=torch.long).to(device)
    out_tensor = model.generate(in_tensor)
    print(" ".join(data.decode(out_tensor[0].tolist())))

def test_modules(config : Config):
    b, t, d = 4, 8, 32
    head_size = d // config.n_heads
    module = tm.AttentionHead(config, head_size)
    in_tensor = torch.zeros((4, 8, 32))
    out_tensor = module(in_tensor)
    out_shape = (4, 8, 8)
    assert out_tensor.shape == out_shape, f"Failed with {out_tensor.shape}"

    b, t, d = 4, 8, 32
    module = tm.MultiHeadAttention(config)
    in_tensor = torch.zeros((4, 8, 32))
    out_tensor = module(in_tensor)
    out_shape = (4, 8, 32)
    assert out_tensor.shape == out_shape, f"Failed with {out_tensor.shape}"

def main():
   config = Config()
   # test_modules(config)
   train_test_model(config)

if __name__ == '__main__':
    main()















