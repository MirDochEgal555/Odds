#!/usr/bin/env python3
import os, json, re, csv, random, pathlib, textwrap, argparse, urllib.request
from datetime import datetime

# ---------------- paths & storage ----------------
ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
FEEDBACK_CSV = DATA / "feedback.csv"   # ts,variant,text,rating,reason
SEEN_JSON = DATA / "seen.json"
BLOCKED_CSV = DATA / "blocked.csv"
WEIGHTS_JSON = DATA / "weights.json"   # bandit over variants
REWARD_FILE = DATA / "reward.joblib"   # sklearn reward model
REWARD_VOCAB = DATA / "reward_vocab.json"

# ---------------- local LLM settings (Ollama) ----------------
GEN_MODEL = os.getenv("ODDS_GEN", "mistral")       # generation model
JUDGE_MODEL = os.getenv("ODDS_JUDGE", "llama3.2:3b")  # optional LLM-as-judge

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# ---------------- defaults (CLI-overridable) ----------------
DEFAULTS = dict(
    language="en",
    vibe="chaotic",            # chill | silly | chaotic
    safety="Dirty",             # PG | PG13 | Party | Dirty
    time=10,                  # minutes
    location="spain, outdoor, after dinner",
    names="Robin,Philipp,Marvin,Daniel,Lex,Tiberio,Sven,Leon,Lorenz",
    keywords="friendly",
)

# ---------------- prompt variants ----------------
PROMPT_BASE = """You design dares for the party game "What Are The Odds?".
Constraints:
- Tone: {vibe}. Safety: {safety}. Language: {language}. Keywords: {keywords}.
- Doable in ≤ {time} minutes, location: ({location}).
- No illegal, medical, hate, bullying.
- Single sentence, concrete action, funny and friendly.
- Remember these shall be dares, which you are not willing to do.
- You shall mention a random player name from: {names}.
Return a JSON array of exactly 30 distinct dares starting with "[name] What are the odds you ..." (strings only)."""

PROMPT_SPICY = "Favor dares which need more willingness and are harder to do. These shall be more spicy and can involve interactions with strangers."

VARIANTS = [
    #("v1_base", PROMPT_BASE, 0.9),
    ("v4_spicy", PROMPT_BASE + "\n\n" + PROMPT_SPICY, 0.9)
]

# --- reward model cache (speeds up + avoids repeated loads)
REWARD_MODEL = None
REWARD_VOCAB_OBJ = None
REWARD_COL_INDEX = None
REWARD_IDF = None
REWARD_NFEATS = None

# ---------------- tiny HTTP helpers ----------------
def _ollama_chat(model: str, messages: list, temperature=0.8):
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        method="POST",
        headers={"Content-Type":"application/json"},
        data=json.dumps({
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }).encode("utf-8")
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def ollama_json_list(model: str, prompt: str, temperature=0.8):
    """Ask the model to return a JSON array; robustly parse."""
    print("ollama_json_list started at: ", datetime.utcnow().isoformat())
    sys = {"role":"system","content":"Return strictly a JSON array of strings. No prose, no preface."}
    out = _ollama_chat(model, [sys, {"role":"user","content":prompt}], temperature)
    content = out["message"]["content"]
    # extract first [...] block
    m = re.search(r"\[.*\]", content, flags=re.S)
    if not m: 
        # fall back: split lines
        items = [s.strip("-• \n\t") for s in content.strip().splitlines() if s.strip()]
        print("ollama_json_list ended at: ", datetime.utcnow().isoformat())
        return [s for s in items if len(s) > 0][:30]
    try:
        arr = json.loads(m.group(0))
        print("ollama_json_list ended at: ", datetime.utcnow().isoformat())
        return [s for s in arr if isinstance(s, str)]
    except Exception:
        print("ollama_json_list ended at: ", datetime.utcnow().isoformat())
        return []

# ---------------- persistence ----------------
def load_seen():
    if SEEN_JSON.exists():
        try: return set(json.loads(SEEN_JSON.read_text()))
        except Exception: pass
    return set()

def save_seen(seen: set, cap=600):
    arr = list(seen)
    if len(arr) > cap: arr = arr[-cap:]
    SEEN_JSON.write_text(json.dumps(arr))

def load_weights():
    if WEIGHTS_JSON.exists():
        return json.loads(WEIGHTS_JSON.read_text())
    return {name: {"mean":0.0, "count":1.0} for name,_,_ in VARIANTS}

def save_weights(w): WEIGHTS_JSON.write_text(json.dumps(w, indent=2))

def append_feedback(variant, text, rating, reason=""):
    newfile = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if newfile: wr.writerow(["ts","variant","text","rating","reason"])
        wr.writerow([datetime.utcnow().isoformat(), variant, text, rating, reason])

def update_weights(w, variant, rating):
    reward = {2:+2.0, 1:+1.0, 0:0.0, -1:-0.5, -2:-1.5}[rating]
    rec = w[variant]; m, n = rec["mean"], rec["count"]
    rec["mean"] = (m*n + reward) / (n+1.0)
    rec["count"] = n + 1.0

# ---------------- safety ----------------
def load_block_patterns():
    pats = []
    if BLOCKED_CSV.exists():
        for line in BLOCKED_CSV.read_text().splitlines():
            s=line.strip()
            if s and not s.startswith("#"): pats.append(re.compile(s, re.I))
    default = [
        #r"\b(drug|alcohol|shots?|beer|wine|vodka)\b",
        #r"\b(sex|naked|strip|kiss|make out)\b",
        r"\b(hate|slur|racist|homophob|ableis|sexist)\b",
        r"\b(steal|trespass|shoplift|vandal)\b",
        r"\b(self[- ]?harm|harm yourself|cut yourself)\b",
        #r"\bmessage|dm|call\b.*\b(stranger|random)\b",
        #r"\bmedical advice\b",
        #r"\bcredit card|password\b",
    ]
    return [re.compile(p, re.I) for p in default] + pats

BLOCKERS = load_block_patterns()
def is_safe(t: str) -> bool:
    return not any(p.search(t) for p in BLOCKERS)

# ---------------- heuristic judge + (optional) LLM judge ----------------
def heuristic_score(text: str, vibe: str, names_list):
    s = 0.0
    L = len(text)
    if 35 <= L <= 120: s += 1.0
    if re.search(r"\b(pretend|set|send|say|act|balance|rename|whisper|draw|mime|hum|describe|guess)\b", text, re.I): s += 0.5
    if vibe == "chill" and re.search(r"\bshout|yell\b", text, re.I): s -= 0.5
    if "what are the odds" in text.lower(): s += 0.5
    #if "if you're up for it" in text.lower(): s += 0.1
    if any(n in text for n in names_list): s += 0.3
    return s

def llm_judge_score(text: str):
    return 0.5
    """Optional local LLM judge, returns 0..1 score."""
    rubric = (
        "Score 0..1 for: clarity, inclusive fun, originality, safety. "
        "Return only a number between 0 and 1."
    )
    out = _ollama_chat(JUDGE_MODEL, [
        {"role":"system","content":rubric},
        {"role":"user","content":f'Dare: "{text}"\nScore:'}
    ], temperature=0.0)
    content = out["message"]["content"].strip()
    try:
        val = float(re.findall(r"0?\.\d+|1(?:\.0+)?", content)[0])
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5

# ---------------- reward model (train from your ratings) ----------------
def load_reward_cache():
    """Load reward model & vocab once into globals."""
    global REWARD_MODEL, REWARD_VOCAB_OBJ, REWARD_COL_INDEX, REWARD_IDF, REWARD_NFEATS
    if not (REWARD_FILE.exists() and REWARD_VOCAB.exists()):
        return
    if REWARD_MODEL is not None:
        return
    from joblib import load
    REWARD_MODEL = load(REWARD_FILE)
    REWARD_VOCAB_OBJ = json.loads(REWARD_VOCAB.read_text())
    order = REWARD_VOCAB_OBJ["__order__"]          # list of tokens in fixed order
    REWARD_COL_INDEX = {tok: i for i, tok in enumerate(order)}
    REWARD_IDF = REWARD_VOCAB_OBJ.get("__idf__", {})
    REWARD_NFEATS = len(order)

def reward_predict(texts):
    """
    Return scores from the ridge reward model, or 0.5 if model not trained.
    Fixes the dimension bug by using len(__order__) for feature count.
    """
    load_reward_cache()
    if REWARD_MODEL is None or REWARD_VOCAB_OBJ is None:
        return [None] * len(texts)

    import numpy as np
    # if the trained model expects a different number of features, bail safely
    try:
        expected = getattr(REWARD_MODEL, "n_features_in_", REWARD_NFEATS)
    except Exception:
        expected = REWARD_NFEATS
    if expected != REWARD_NFEATS:
        # mismatched artifacts (old model with new vocab). Ask user to retrain.
        print("[reward] Feature mismatch between model and vocab — run `train-reward` again.")
        return [0.5] * len(texts)

    rows = np.zeros((len(texts), REWARD_NFEATS), dtype="float32")

    token_re = re.compile(r"[a-zA-Z]{2,}")
    for r, t in enumerate(texts):
        toks = token_re.findall(t.lower())
        if not toks:
            continue
        # term counts limited to known vocab
        counts = {}
        for tok in toks:
            if tok in REWARD_COL_INDEX:
                counts[tok] = counts.get(tok, 0) + 1
        doclen = sum(counts.values()) or 1
        for tok, c in counts.items():
            j = REWARD_COL_INDEX[tok]
            tf = c / doclen
            rows[r, j] = tf * float(REWARD_IDF.get(tok, 1.0))

    preds = REWARD_MODEL.predict(rows)
    return preds.tolist()

def train_reward():
    """Train a tiny TF-IDF + Ridge regressor from feedback.csv (fully local)."""
    if not FEEDBACK_CSV.exists():
        print("No feedback yet.")
        return
    rows = []
    with open(FEEDBACK_CSV, newline="", encoding="utf-8") as f:
        for i,row in enumerate(csv.DictReader(f)):
            try:
                r = int(row["rating"])
            except Exception:
                continue
            txt = row["text"].strip()
            if not txt: continue
            # map ratings to targets (skip = -1 weaker than meh=0)
            target = {2:1.0, 1:0.6, 0:0.3, -1:0.15, -2:0.0}.get(r, 0.3)
            rows.append((txt, target))
    if len(rows) < 30:
        print("Need at least ~30 rated dares; keep playing a bit more.")
        return
    texts, y = zip(*rows)
    # build vocab (pruned)
    def tokenize(t):
        return re.findall(r"[a-zA-Z]{2,}", t.lower())
    from collections import Counter
    df = Counter()
    for t in set(texts):
        toks = set(tokenize(t))
        for tok in toks: df[tok]+=1
    keep = [tok for tok,c in df.items() if 2 <= c <= len(texts)*0.6][:5000]
    order = sorted(keep)
    # idf
    import math
    N = len(texts)
    idf = {tok: math.log((N+1)/(df[tok]+1)) + 1.0 for tok in order}
    # transform
    import numpy as np
    X = np.zeros((len(texts), len(order)), dtype="float32")
    for i,t in enumerate(texts):
        toks = tokenize(t)
        cnt = Counter([tok for tok in toks if tok in idf])
        L = sum(cnt.values()) or 1
        for tok,c in cnt.items():
            X[i, order.index(tok)] = (c/L)*idf[tok]
    # fit ridge
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.8)
    model.fit(X, y)
    from joblib import dump
    dump(model, REWARD_FILE)
    REWARD_VOCAB.write_text(json.dumps({"__order__":order, "__idf__":idf}))
    print(f"Trained reward model on {len(texts)} examples → {REWARD_FILE}")

# ---------------- bandit & generation ----------------
def thompson_pick(weights):
    best, name = None, None
    for k,v in weights.items():
        mean, count = v["mean"], max(v["count"],1.0)
        sample = random.gauss(mean, (1.0/count)**0.5)
        if best is None or sample > best:
            best, name = sample, k
    return name

def render(tmpl: str, ctx: dict) -> str: return tmpl.format(**ctx)

def generate_candidates(ctx, variant_name, tmpl, temp):
    print("generate_candidates started at: ", datetime.utcnow().isoformat())
    prompt = render(tmpl, ctx)
    items = ollama_json_list(GEN_MODEL, prompt, temperature=temp)
    out = []
    for s in items:
        s = s.strip()
        if s and is_safe(s): out.append(s)
    # dedupe with seen
    seen = load_seen()
    out = [s for s in out if s.lower() not in seen]
    if not out:
        # fallback: allow seen
        out = [s.strip() for s in items if s.strip() and is_safe(s)]
    print("generate_candidates ended at: ", datetime.utcnow().isoformat())
    return out

def pick_best(cands, ctx):
    """Combine reward model, heuristic, and optional LLM judge."""
    if not cands: return None
    # 1) reward model (trained from your ratings)
    r_preds = reward_predict(cands)
    have_reward = any(p is not None for p in r_preds)
    # 2) heuristic
    h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in cands]
    # 3) optional LLM judge (very small model)
    llm_scores = []
    try:
        llm_scores = [llm_judge_score(t) for t in cands]
    except Exception:
        llm_scores = [0.5]*len(cands)
    # fuse
    fused = []
    for i,t in enumerate(cands):
        rp = r_preds[i] if have_reward else 0.5
        hs = h_scores[i]
        ls = llm_scores[i]
        # weighted sum (reward dominates when available)
        score = (0.55*rp + 0.25*ls + 0.20*(hs/2.0))
        fused.append((score, t))
    fused.sort(reverse=True, key=lambda x:x[0])
    return fused[0][1]

def generate_one(ctx, weights):
    variant = thompson_pick(weights)
    tmpl, temp = None, 0.9
    for name, T, temperature in VARIANTS:
        if name == variant:
            tmpl, temp = T, temperature
            break
    cands = generate_candidates(ctx, variant, tmpl, temp)
    if not cands:
        return variant, "No safe candidates; adjust filters."
    chosen = pick_best(cands, ctx)
    return variant, chosen

# ------- ranking helper (reuse your fusion logic) -------
def rank_candidates_scored(cands_with_variant, ctx):
    """cands_with_variant: List[(variant, text)] -> List[(score, variant, text)]"""
    print("rank_candidates_scored started at: ", datetime.utcnow().isoformat())
    if not cands_with_variant:
        return []
    texts = [t for _, t in cands_with_variant]
    # reward scores (0..1), or 0.5 if not trained yet
    r_preds = reward_predict(texts)
    have_reward = any(p is not None for p in r_preds)

    # heuristic + optional LLM judge
    h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in texts]
    try:
        l_scores = [llm_judge_score(t) for t in texts]
    except Exception:
        l_scores = [0.5] * len(texts)

    ranked = []
    for i, (variant, text) in enumerate(cands_with_variant):
        rp = r_preds[i] if have_reward else 0.5
        hs = h_scores[i]
        ls = l_scores[i]
        score = (0.55 * rp + 0.25 * ls + 0.20 * (hs / 2.0))
        ranked.append((score, variant, text))
    ranked.sort(reverse=True, key=lambda x: x[0])
    print("rank_candidates_scored ended at: ", datetime.utcnow().isoformat())
    return ranked

# ------- NEW: get the top-N dares across variants -------
def get_top_dares(ctx, top_n=25, per_variant_limit=12, include_seen=False):
    """
    Generate candidates from all prompt variants, rank them, and return the best top_n.
    - Respects your safety filters and (by default) your seen history.
    - per_variant_limit is controlled inside generate_candidates() via the prompt size,
      but we still call each variant once and then merge+rank here.
    """
    print("get_top_dares started at: ", datetime.utcnow().isoformat())
    pool = []
    for name, tmpl, temp in VARIANTS:
        cands = generate_candidates(ctx, name, tmpl, temp)
        # generate_candidates already avoids seen; optionally allow seen:
        if include_seen:
            # pull again without dedup if pool ended up too small
            if len(cands) < top_n:
                raw = ollama_json_list(GEN_MODEL, render(tmpl, ctx), temperature=temp)
                more = [s.strip() for s in raw if isinstance(s, str) and s.strip() and is_safe(s)]
                cands = list({*cands, *more})  # quick union dedupe
        pool.extend((name, t) for t in cands)

    # de-dup across variants
    seen_texts = set()
    uniq = []
    for variant, text in pool:
        key = text.lower().strip()
        if key not in seen_texts:
            seen_texts.add(key)
            uniq.append((variant, text))

    print("Printing uniq:")
    for i in uniq:
        print(i)
    ranked = rank_candidates_scored(uniq, ctx)
    print("Printing ranked:")
    for i in ranked:
        print(i)
    print("get_top_dares ended at: ", datetime.utcnow().isoformat())
    return ranked[:top_n]  # [(score, variant, text)]


# ---------------- batch generation (for "make-pack") ----------------
def batch_generate(ctx, weights, total=150, allow_seen=False, seed=None):
    """
    Generate N dares using local generator + reward model ranking.
    - Honors safety filters.
    - Dedupes within the batch and (by default) against seen.json unless allow_seen=True.
    """
    if seed is not None:
        random.seed(seed)

    # prefer better variants via softmax over means
    import math
    names = list(weights.keys())
    means = [weights[k]["mean"] for k in names]
    mx = max(means) if means else 0.0
    probs = [math.exp(m - mx) for m in means]
    seen_global = load_seen()
    bag, attempts = [], 0
    cap_attempts = total * 10  # safety cap

    while len(bag) < total and attempts < cap_attempts:
        attempts += 1
        print(f"[progress] {len(bag)}/{total} dares collected after {attempts} attempts")
        variant = random.choices(names, weights=probs, k=1)[0]
        # find variant template
        tmpl, temp = None, 0.9
        for name, T, temperature in VARIANTS:
            if name == variant:
                tmpl, temp = T, temperature
                break
        # get candidates
        cands = generate_candidates(ctx, variant, tmpl, temp)
        if not cands:
            continue
        # rank candidates with reward/heuristic/LLM judge
        ranked = []
        r_preds = reward_predict(cands)
        have_reward = any(p is not None for p in r_preds)
        h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in cands]
        try:
            l_scores = [llm_judge_score(t) for t in cands]
        except Exception:
            l_scores = [0.5] * len(cands)
        for i,t in enumerate(cands):
            rp = r_preds[i] if have_reward else 0.5
            hs = h_scores[i]
            ls = l_scores[i]
            score = (0.55*rp + 0.25*ls + 0.20*(hs/2.0))
            ranked.append((score, variant, t))
        ranked.sort(reverse=True, key=lambda x: x[0])
        print(f"[debug] Got {len(ranked)} candidates from {variant}, best score={ranked[0][0]:.3f}")

        # take top few each loop
        for _, var, t in ranked[:15]:
            key = t.lower()
            if not allow_seen and key in seen_global:
                continue
            if all(key != x.lower() for x in bag):
                bag.append(t)
            if len(bag) >= total:
                break

    return bag

# ---------------- CLI add-on for make-pack ----------------
def make_pack_cmd(ctx, weights, total=150, out_file="vacation_pack.md", allow_seen=False, seed=None):
    items = batch_generate(ctx, weights, total=total, allow_seen=allow_seen, seed=seed)
    md = [
        "# What Are The Odds? — Custom Pack",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        f"_Vibe: {ctx['vibe']} | Safety: {ctx['safety']} | Time ≤ {ctx['time']} min_",
        ""
    ]
    for i, t in enumerate(items, 1):
        md.append(f"{i}. {t}")
    text = "\n".join(md)
    (ROOT / out_file).write_text(text, encoding="utf-8")
    # optional PDF via pandoc if installed
    try:
        os.system(f"pandoc {out_file} -o {out_file.replace('.md', '.pdf')} >/dev/null 2>&1")
    except Exception:
        pass
    print(f"Saved {len(items)} dares → {out_file}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="What Are The Odds? — local, dare-level learning")
    ap.add_argument("cmd", choices=["play","train-reward","make-pack","play-top"],
                    help="play = one dare; train-reward = fit local scorer; make-pack = batch export")
    ap.add_argument("--language", default=DEFAULTS["language"])
    ap.add_argument("--vibe", default=DEFAULTS["vibe"], choices=["chill","silly","chaotic"])
    ap.add_argument("--safety", default=DEFAULTS["safety"], choices=["PG","PG13","Party","Dirty"])
    ap.add_argument("--time", type=int, default=DEFAULTS["time"])
    ap.add_argument("--location", default=DEFAULTS["location"])
    ap.add_argument("--names", default=DEFAULTS["names"], help="Comma-separated names")
    ap.add_argument("--auto", action="store_true", help="Non-interactive: auto rate 0")
    # new flags for make-pack
    ap.add_argument("--total", type=int, default=150, help="How many dares to export (make-pack)")
    ap.add_argument("--out", type=str, default="vacation_pack.md", help="Output markdown file (make-pack)")
    ap.add_argument("--allow-seen", action="store_true", help="Allow repeats from your seen history (make-pack)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (make-pack)")
    ap.add_argument("--top-n", type=int, default=25, help="How many top dares to show in play-top")
    ap.add_argument("--keywords", default=DEFAULTS["keywords"], help="Extra Keywords")

    args = ap.parse_args()

    if args.cmd == "train-reward":
        train_reward()
        return

    ctx = {
        "language": args.language,
        "vibe": args.vibe,
        "safety": args.safety,
        "time": args.time,
        "location": args.location,
        "names": args.names,
        "names_list": [n.strip() for n in args.names.split(",") if n.strip()],
        "keywords": args.keywords
    }
    weights = load_weights()

    load_reward_cache()

    if args.cmd == "make-pack":
        make_pack_cmd(ctx, weights, total=args.total, out_file=args.out,
                      allow_seen=bool(args.allow_seen), seed=args.seed)
        return

    # play
    ctx = {
        "language": args.language,
        "vibe": args.vibe,
        "safety": args.safety,
        "time": args.time,
        "location": args.location,
        "names": args.names,
        "names_list": [n.strip() for n in args.names.split(",") if n.strip()],
        "keywords": args.keywords
    }

    if args.cmd == "play-top":
        items = get_top_dares(ctx, top_n=args.top_n)
        if not items:
            print("No candidates found. Try relaxing filters or retraining reward.")
            return

        # show all top-N, then rate each
        for idx, (score, variant, dare) in enumerate(items, 1):
            print(f"\n[{idx}/{len(items)}] score={score:.3f}  variant={variant}")
            print("—", textwrap.fill(dare, width=88))
            ans = input("Rate (-2,-1,0,1,2) or Enter=0, q=quit: ").strip().lower()
            if ans == "q":
                break
            rating = int(ans) if ans in {"-2","-1","0","1","2"} else 0
            reason = ""
            if rating == -2:
                reason = input("Reason/block regex (optional): ").strip()
                if reason:
                    with open(BLOCKED_CSV, "a", encoding="utf-8") as f:
                        f.write(reason + "\n")
            append_feedback(variant, dare, rating, reason)
            update_weights(load_weights(), variant, rating)  # reload -> update -> save
            save_seen(load_seen() | {dare.lower()})
        return


    weights = load_weights()
    try:
        variant, dare = generate_one(ctx, weights)
    except Exception as e:
        print("Generation error:", e)
        return

    print("\n— Dare —")
    print(textwrap.fill(dare, width=88))
    print(f"\n(variant: {variant} | gen: {GEN_MODEL} | judge: {JUDGE_MODEL})")

    # seen memory
    seen = load_seen()
    h = dare.lower()
    if h not in seen:
        seen.add(h); save_seen(seen)

    # rating
    if args.auto:
        rating = 0; reason = ""
    else:
        print("\nRate: [2]=love  [1]=like  [0]=meh  [-1]=skip  [-2]=report/block")
        ans = input("Your rating (-2,-1,0,1,2) or Enter for 0: ").strip()
        rating = int(ans) if ans in {"-2","-1","0","1","2"} else 0
        reason = ""
        if rating == -2:
            reason = input("Reason or regex to block (optional): ").strip()
            if reason:
                with open(BLOCKED_CSV, "a", encoding="utf-8") as f:
                    f.write(reason + "\n")

    append_feedback(variant, dare, rating, reason)
    update_weights(weights, variant, rating)
    save_weights(weights)
    print("\nSaved feedback. Weights now:")
    print(json.dumps(weights, indent=2))

if __name__ == "__main__":
    main()
