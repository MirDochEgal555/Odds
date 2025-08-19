#!/usr/bin/env python3
import csv, json, random, math, pathlib
from dataclasses import dataclass
from typing import List, Tuple
import torch, torch.nn as nn, torch.optim as optim
from sentence_transformers import SentenceTransformer
from joblib import dump

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
FEEDBACK = DATA / "feedback.csv"
MODEL_DIR = DATA / "nn_reward"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- Data ----------------
@dataclass
class Example:
    text: str
    rating: int

def load_feedback() -> List[Example]:
    rows = []
    with open(FEEDBACK, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try: rating = int(r["rating"])
            except: continue
            t = (r["text"] or "").strip()
            if not t: continue
            rows.append(Example(t, rating))
    return rows

def make_pairs(ex: List[Example], max_pairs=50000) -> List[Tuple[str,str]]:
    pos = [e.text for e in ex if e.rating >= 1]
    neg = [e.text for e in ex if e.rating <= 0]
    random.shuffle(pos); random.shuffle(neg)
    n = min(len(pos), len(neg), max_pairs)
    return list(zip(pos[:n], neg[:n]))

# ---------------- Model ----------------
class RankMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)  # scalar score
        )
    def forward(self, x):  # x: [B, D]
        return self.net(x).squeeze(-1)  # [B]

def pairwise_margin_loss(pos, neg, margin=0.2):
    # pos, neg: [B] scores
    return torch.clamp(margin - (pos - neg), min=0).mean()

# ---------------- Train ----------------
def train(seed=42, batch_size=64, epochs=6, lr=1e-3, margin=0.2, device="cpu"):
    random.seed(seed); torch.manual_seed(seed)
    ex = load_feedback()
    if len(ex) < 40:
        print("Need ≥ 40 rated dares to start training."); return
    pairs = make_pairs(ex)
    if not pairs:
        print("No positive/negative pairs found."); return
    print(f"Training on {len(pairs)} pairs.")

    # frozen encoder (download once; then fully local)
    enc = SentenceTransformer("all-MiniLM-L6-v2", device=device)  # 384-dim
    enc.eval()
    dim = enc.get_sentence_embedding_dimension()

    mlp = RankMLP(dim).to(device)
    opt = optim.AdamW(mlp.parameters(), lr=lr)
    best_loss, best_state = float("inf"), None

    # simple batching
    def batches(lst, bsz):
        for i in range(0, len(lst), bsz):
            yield lst[i:i+bsz]

    for epoch in range(1, epochs+1):
        random.shuffle(pairs)
        total, steps = 0.0, 0
        for batch in batches(pairs, batch_size):
            pos_texts, neg_texts = zip(*batch)
            with torch.no_grad():
                pos_emb = torch.tensor(enc.encode(list(pos_texts), normalize_embeddings=True)).to(device)
                neg_emb = torch.tensor(enc.encode(list(neg_texts), normalize_embeddings=True)).to(device)
            opt.zero_grad()
            pos_scores = mlp(pos_emb)
            neg_scores = mlp(neg_emb)
            loss = pairwise_margin_loss(pos_scores, neg_scores, margin=margin)
            loss.backward()
            opt.step()
            total += loss.item(); steps += 1
        avg = total / max(steps,1)
        print(f"epoch {epoch}: loss={avg:.4f}")
        if avg < best_loss:
            best_loss, best_state = avg, {k:v.cpu() for k,v in mlp.state_dict().items()}

    # save artifacts
    torch.save(best_state, MODEL_DIR / "rank_mlp.pt")
    # Save a tiny config & encoder name for inference
    (MODEL_DIR / "config.json").write_text(json.dumps({
        "encoder": "all-MiniLM-L6-v2",
        "dim": dim,
        "margin": margin
    }, indent=2))
    print(f"Saved nn reward model → {MODEL_DIR}")

if __name__ == "__main__":
    import argparse, torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.2)
    args = ap.parse_args()
    train(device=args.device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, margin=args.margin)

