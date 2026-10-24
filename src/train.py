
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from model import TinyGPT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "alice.txt"
OUT = ROOT / "checkpoints"
OUT.mkdir(exist_ok=True)

SEED = 42
BLOCK = 128
BATCH = 16
STEPS = 3000
EVAL_EVERY = 250
LR = 3e-4

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

text = DATA.read_text(encoding="utf-8", errors="replace")
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

split = int(0.9 * len(encoded))
train_data, val_data = encoded[:split], encoded[split:]

if len(val_data) <= BLOCK:
    raise ValueError("Corpus is too small for the chosen block size.")

def get_batch():
    source = train_data if random.random() < 0.9 else val_data
    starts = torch.randint(len(source) - BLOCK - 1, (BATCH,))
    x = torch.stack([source[i:i + BLOCK] for i in starts])
    y = torch.stack([source[i + 1:i + BLOCK + 1] for i in starts])
    return x.to(device), y.to(device)

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=6,
    block_size=BLOCK,
    dropout=0.1,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(20):
            starts = torch.randint(len(source) - BLOCK - 1, (BATCH,))
            x = torch.stack([source[i:i + BLOCK] for i in starts]).to(device)
            y = torch.stack([source[i + 1:i + BLOCK + 1] for i in starts]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results

best_val = math.inf

for step in range(STEPS):
    x, y = get_batch()
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        metrics = estimate_loss()
        print(
            f"step {step:5d} | "
            f"train {metrics['train']:.3f} | "
            f"val {metrics['val']:.3f}"
        )

        if metrics["val"] < best_val:
            best_val = metrics["val"]
            torch.save({
                "model": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": 256,
                    "n_heads": 8,
                    "n_layers": 6,
                    "block_size": BLOCK,
                    "dropout": 0.1,
                },
                "step": step,
                "val_loss": best_val,
            }, OUT / "tinygpt.pt")

print("Training finished. Best validation loss:", best_val)
