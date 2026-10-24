
from pathlib import Path
import torch
from model import TinyGPT

ROOT = Path(__file__).resolve().parents[1]
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(
    ROOT / "checkpoints" / "tinygpt.pt",
    map_location=device,
    weights_only=False,
)

cfg = ckpt["config"]
model = TinyGPT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

stoi = ckpt["stoi"]
itos = {int(k): v for k, v in ckpt["itos"].items()}

prompt = input("Starting text: ")
unknown = [c for c in prompt if c not in stoi]
if unknown:
    print("Unknown characters will be skipped:", repr("".join(unknown)))
prompt = "".join(c for c in prompt if c in stoi)

if not prompt:
    prompt = "\n"

idx = torch.tensor(
    [[stoi[c] for c in prompt]], dtype=torch.long, device=device
)

with torch.no_grad():
    result = model.generate(idx, max_new_tokens=300, temperature=0.8)

print("Generated text:")
print("".join(itos[i] for i in result[0].tolist()))
