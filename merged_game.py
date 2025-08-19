
#!/usr/bin/env python3
# merged_game.py
# Combines the generation/learning engine (from app.py) with the Tk UI (from UI.py)
# to provide a single runnable program.

import os, json, re, csv, random, pathlib, textwrap, argparse, urllib.request, sys, math, time
from datetime import datetime
import threading
from tkinter import ttk, messagebox, simpledialog


# ===============================
# === Engine (from app.py) ======
# ===============================

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

# ---------------- defaults (UI will override) ----------------
DEFAULTS = dict(
    language="en",
    vibe="chaotic",            # chill | silly | chaotic
    safety="Party",            # PG | PG13 | Party
    time=10,                   # minutes
    location="spain, outdoor, after dinner",
    names="Robin,Philipp,Marvin,Daniel,Lex,Tiberio,Sven,Leon,Lorenz",
    keywords="friendly",
)

PROMPT_BASE = """You design dares for the party game "What Are The Odds?".
Constraints:
- Tone: {vibe}. Safety: {safety}. Language: {language}. Setting: {keywords}.
- Doable in ≤ {time} minutes, location: ({location}).
- No illegal, medical, hate, bullying.
- Single sentence, concrete action, funny and friendly.
- Remember these shall be dares, which you are not willing to do.
- You shall mention a random player name from: {names}.
Return a JSON array of exactly 55 distinct dares starting with "[name] What are the odds you ..." (strings only)."""

PROMPT_SPICY = "Favor dares which need more willingness and are harder to do. These shall be more spicy and can involve interactions with strangers."

VARIANTS = [
    ("v4_spicy", PROMPT_BASE + "\n\n" + PROMPT_SPICY, 0.9)
]

# --- reward model cache (speeds up + avoids repeated loads)
REWARD_MODEL = None
REWARD_VOCAB_OBJ = None
REWARD_COL_INDEX = None
REWARD_IDF = None
REWARD_NFEATS = None

# ---------------- Novelty score helpers -----------
from collections import Counter

WORD_FREQ_JSON = DATA / "word_freq.json"
NOVELTY_ALPHA = float(os.getenv("ODDS_NOVELTY", "0.15"))  # 0 disables novelty

def load_word_freq():
    if WORD_FREQ_JSON.exists():
        try:
            return Counter(json.loads(WORD_FREQ_JSON.read_text()))
        except Exception:
            pass
    return Counter()

def save_word_freq(freqs: Counter):
    WORD_FREQ_JSON.write_text(json.dumps(freqs, indent=2))

# very small stoplist + game boilerplate words we don't want to count
STOPWORDS = set("""
a an the and or to of in on for with as is are be you your we us our at by from it this that these those
what odds odds? odds! odds. are the you what are the odds you
""".split())

_token_re = re.compile(r"[a-z]{3,}")

def tokenize_words(t: str):
    toks = [w for w in _token_re.findall(t.lower()) if w not in STOPWORDS]
    return toks

def novelty_score(text: str, freqs: Counter, sat: int = 8) -> float:
    """
    Average 'rarity' of the tokens in text.
    For each token: score = sat / (freq + sat)  -> unseen=1.0, freq>>sat ~ 0
    Returns 0..1 (higher = more novel).
    """
    toks = tokenize_words(text)
    if not toks:
        return 0.0
    vals = [sat / (freqs.get(tok, 0) + sat) for tok in toks]
    return sum(vals) / len(vals)

def update_word_freqs(text: str, freqs: Counter):
    for tok in tokenize_words(text):
        freqs[tok] += 1


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
    sys_msg = {"role":"system","content":"Return strictly a JSON array of strings. No prose, no preface."}
    out = _ollama_chat(model, [sys_msg, {"role":"user","content":prompt}], temperature)
    content = out["message"]["content"]
    # extract first [...] block
    m = re.search(r"\[.*\]", content, flags=re.S)
    if not m: 
        # fall back: split lines
        items = [s.strip("-• \n\t") for s in content.strip().splitlines() if s.strip()]
        print("ollama_json_list ended at: ", datetime.utcnow().isoformat())
        return [s for s in items if len(s) > 0][:55]
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
    save_weights(w)

# ---------------- safety ----------------
def load_block_patterns():
    pats = []
    if BLOCKED_CSV.exists():
        for line in BLOCKED_CSV.read_text().splitlines():
            s=line.strip()
            if s and not s.startswith("#"): pats.append(re.compile(s, re.I))
    default = [
        r"\b(hate|slur|racist|homophob|ableis|sexist)\b",
        r"\b(steal|trespass|shoplift|vandal)\b",
        r"\b(self[- ]?harm|harm yourself|cut yourself)\b",
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
    if any(n in text for n in names_list): s += 0.3
    #if re.search(r"\b")
    return s

def llm_judge_score(text: str):
    # keep it simple/fast by default
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
    """Combine reward model, heuristic, optional LLM judge, and novelty bonus."""
    if not cands:
        return None
    r_preds = reward_predict(cands)
    have_reward = any(p is not None for p in r_preds)
    h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in cands]
    try:
        llm_scores = [llm_judge_score(t) for t in cands]
    except Exception:
        llm_scores = [0.5] * len(cands)

    # --- NEW: novelty per candidate; pull freqs from a module-level cache if present
    # Fallback to empty if Engine hasn't loaded yet (e.g., in a unit test)
    try:
        freqs = ENGINE.word_freqs  # populated in Engine.__init__
    except Exception:
        freqs = Counter()

    novelty_scores = [novelty_score(t, freqs) for t in cands]

    fused = []
    for i, t in enumerate(cands):
        rp = r_preds[i] if have_reward else 0.5
        hs = h_scores[i]
        ls = llm_scores[i]
        ns = novelty_scores[i]  # 0..1
        base = (0.55 * rp + 0.25 * ls + 0.20 * (hs / 2.0))
        score = base + NOVELTY_ALPHA * ns
        fused.append((score, t))
    fused.sort(reverse=True, key=lambda x: x[0])
    return fused[0][1]

def rank_candidates_scored(cands_with_variant, ctx):
    print("rank_candidates_scored started at: ", datetime.utcnow().isoformat())
    if not cands_with_variant:
        return []
    texts = [t for _, t in cands_with_variant]
    r_preds = reward_predict(texts)
    have_reward = any(p is not None for p in r_preds)
    h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in texts]
    try:
        l_scores = [llm_judge_score(t) for t in texts]
    except Exception:
        l_scores = [0.5] * len(texts)

    try:
        freqs = ENGINE.word_freqs
    except Exception:
        freqs = Counter()
    n_scores = [novelty_score(t, freqs) for t in texts]

    ranked = []
    for i, (variant, text) in enumerate(cands_with_variant):
        rp = r_preds[i] if have_reward else 0.5
        hs = h_scores[i]
        ls = l_scores[i]
        ns = n_scores[i]
        base = (0.55 * rp + 0.25 * ls + 0.20 * (hs / 2.0))
        score = base + NOVELTY_ALPHA * ns
        ranked.append((score, variant, text))
    ranked.sort(reverse=True, key=lambda x: x[0])
    print("rank_candidates_scored ended at: ", datetime.utcnow().isoformat())
    return ranked


# ------- NEW: get the top-N dares across variants -------
def get_top_dares(ctx, top_n=25, per_variant_limit=12, include_seen=False):
    """
    Generate candidates from all prompt variants, rank them, and return the best top_n.
    Returns: [(score, variant, text)]
    """
    print("get_top_dares started at: ", datetime.utcnow().isoformat())
    pool = []
    for name, tmpl, temp in VARIANTS:
        cands = generate_candidates(ctx, name, tmpl, temp)
        if include_seen and len(cands) < top_n:
            raw = ollama_json_list(GEN_MODEL, render(tmpl, ctx), temperature=temp)
            more = [s.strip() for s in raw if isinstance(s, str) and s.strip() and is_safe(s)]
            cands = list({*cands, *more})  # union dedupe
        pool.extend((name, t) for t in cands)

    # de-dup across variants
    seen_texts = set()
    uniq = []
    for variant, text in pool:
        key = text.lower().strip()
        if key not in seen_texts:
            seen_texts.add(key)
            uniq.append((variant, text))

    ranked = rank_candidates_scored(uniq, ctx)
    print("get_top_dares ended at: ", datetime.utcnow().isoformat())
    return ranked[:top_n]  # [(score, variant, text)]


def thompson_pick(weights):
    best, name = None, None
    for k,v in weights.items():
        mean, count = v["mean"], max(v["count"],1.0)
        sample = random.gauss(mean, (1.0/count)**0.5)
        if best is None or sample > best:
            best, name = sample, k
    return name

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

# A small utility to turn players list into ctx.names and names_list
def ctx_from_players(players, prev_ctx=None, keywords=""):
    if prev_ctx is None:
        prev_ctx = DEFAULTS.copy()
    names = ", ".join(players) if players else prev_ctx.get("names", DEFAULTS["names"])
    base = {
        "language": prev_ctx.get("language", "en"),
        "vibe": prev_ctx.get("vibe", "chaotic"),
        "safety": prev_ctx.get("safety", "Party"),
        "time": prev_ctx.get("time", 10),
        "location": prev_ctx.get("location", "spain, outdoor, after dinner"),
        "names": names,
        "names_list": [n.strip() for n in names.split(",") if n.strip()],
        "keywords": keywords if keywords else prev_ctx.get("keywords", ""),
    }

    return base

class Engine:
    """
    Minimal adapter around the app.py engine so the UI can call:
    - start_new_game()
    - save_players(players)
    - get_next_question() -> {id, text}
    - submit_keywords(round_number, question_id, keywords, players)
    - mark_round_started(round_number, question_id, players, keywords=None)
    - submit_feedback(round_number, question_id, rating, players, keywords=None)
    """
    def __init__(self):
        self.players = []
        self.round = 1
        self.last_keywords = ""
        self.weights = load_weights()
        self.ctx = ctx_from_players(self.players, DEFAULTS, "")
        for n in self.ctx["names_list"]:
            STOPWORDS.add(n.lower())
        self.word_freqs = load_word_freq()
        self._qid_counter = 0
                # --- prefetch/progress state for the loading screen ---
        self._fetch_lock = threading.Lock()
        self._fetch_status = {"loaded": 0, "total": 0, "percent": 0.0,
                              "message": "", "done": True, "error": None}
        self._prefetched = []
        self._fetch_thread = None


    def start_new_game(self):
        self.round = 1
        self._qid_counter = 0
        print("[Engine] New game started.")

    def save_players(self, players):
        self.players = list(players)
        self.ctx = ctx_from_players(self.players, self.ctx, self.last_keywords)
        print(f"[Engine] Players saved: {self.players}")
    
    def begin_fetch_questions(self, total: int = 1, kw_list=None, kw_quota: int = 0):
        """Start background prefetch of `total` questions; try to meet kw_quota with kw_list."""
        with self._fetch_lock:
            if self._fetch_thread is not None and self._fetch_thread.is_alive():
                return  # already running
            total = max(1, int(total or 1))
            kw_list = [k.lower() for k in (kw_list or [])]
            kw_quota = max(0, min(total, int(kw_quota or 0)))
            self._fetch_status = {"loaded": 0, "total": total, "percent": 0.0,
                                "message": "Starting…", "done": False, "error": None}

        def contains_kw(text: str) -> bool:
            if not kw_list:
                return False
            t = (text or "").lower()
            return any(k in t for k in kw_list)

        def _worker():
            try:
                local_ctx_kw = ctx_from_players(self.players, self.ctx, ", ".join(kw_list) if kw_list else "")
                local_ctx_no = ctx_from_players(self.players, self.ctx, "")  # no keywords

                ranked_kw = []
                ranked_no = []
                try:
                    # Fetch a little extra to survive filtering
                    want_kw = kw_quota if kw_quota > 0 else 0
                    want_no = total - want_kw
                    if want_kw > 0:
                        ranked_kw = get_top_dares(local_ctx_kw, top_n=max(want_kw * 2, want_kw + 5), include_seen=False)
                    if want_no > 0:
                        ranked_no = get_top_dares(local_ctx_no, top_n=max(want_no * 2, want_no + 5), include_seen=False)
                except Exception as e:
                    print("[Engine] get_top_dares error, falling back:", e)
                    ranked_kw, ranked_no = [], []

                # Build selection
                selected = []
                seen_texts = set()

                # 1) take up to kw_quota that actually contain a keyword
                for _, variant, text in ranked_kw:
                    if len(selected) >= kw_quota:
                        break
                    if not text:
                        continue
                    key = text.lower().strip()
                    if key in seen_texts:
                        continue
                    if contains_kw(text):
                        selected.append((variant, text))
                        seen_texts.add(key)

                # 2) fill remainder from NON-keyword pool
                for _, variant, text in ranked_no:
                    if len(selected) >= total:
                        break
                    if not text:
                        continue
                    key = text.lower().strip()
                    if key in seen_texts:
                        continue
                    selected.append((variant, text))
                    seen_texts.add(key)

                # 3) still short? take more from KW pool even if they don't literally contain the keyword
                if len(selected) < total:
                    for _, variant, text in ranked_kw:
                        if len(selected) >= total:
                            break
                        if not text:
                            continue
                        key = text.lower().strip()
                        if key in seen_texts:
                            continue
                        selected.append((variant, text))
                        seen_texts.add(key)

                # 4) last resort fallback: use get_next_question() loop
                while len(selected) < total:
                    q = self.get_next_question()
                    if not q or not q.get("text"):
                        break
                    variant = getattr(self, "_last_variant", "v4_spicy")
                    text = q["text"]
                    key = text.lower().strip()
                    if key in seen_texts:
                        continue
                    selected.append((variant, text))
                    seen_texts.add(key)

                # Materialize into prefetched queue and update progress status
                for variant, text in selected[:total]:
                    # mark seen
                    seen = load_seen()
                    key = text.lower()
                    if key not in seen:
                        seen.add(key); save_seen(seen)
                    # build question
                    self._qid_counter += 1
                    qid = f"q{self._qid_counter}_{int(time.time())}"
                    with self._fetch_lock:
                        self._prefetched.append({"id": qid, "text": text, "variant": variant})
                        self._fetch_status["loaded"] += 1
                        t = self._fetch_status["total"]
                        self._fetch_status["percent"] = (self._fetch_status["loaded"] * 100.0) / t
                        self._fetch_status["message"] = f"Fetched {self._fetch_status['loaded']}/{t}"

            except Exception as e:
                with self._fetch_lock:
                    self._fetch_status["error"] = str(e)
            finally:
                with self._fetch_lock:
                    self._fetch_status["done"] = True

        t = threading.Thread(target=_worker, daemon=True)
        self._fetch_thread = t
        t.start()


    def get_fetch_status(self):
        """Return a snapshot dict with keys: loaded,total,percent,message,done,error."""
        with self._fetch_lock:
            return dict(self._fetch_status)

    def consume_prefetched_question(self):
        """Pop one prefetched question if available and set last_* for feedback."""
        with self._fetch_lock:
            if self._prefetched:
                item = self._prefetched.pop(0)
            else:
                item = None
        if item:
            # keep last_* in sync so submit_feedback uses the correct variant/text
            self._last_text = item.get("text")
            self._last_variant = item.get("variant", "v4_spicy")
            return {"id": item.get("id"), "text": item.get("text")}
        return None



    def get_next_question(self):
        """
        Generates a new dare (question) using current players and last keywords.
        Returns a dict with 'id' and 'text'.
        """
        # fold round keywords into ctx for generation
        local_ctx = dict(self.ctx)
        local_ctx["keywords"] = self.last_keywords or self.ctx.get("keywords", "")
        try:
            variant, dare = generate_one(local_ctx, self.weights)
        except Exception as e:
            print("[Engine] Generation error:", e)
            return None
        if not dare:
            return None
        # mark as seen
        seen = load_seen()
        key = dare.lower()
        if key not in seen:
            seen.add(key); save_seen(seen)
        # create a simple id
        self._qid_counter += 1
        qid = f"q{self._qid_counter}_{int(time.time())}"
        print(f"[Engine] get_next_question -> {qid}: {dare} (variant={variant})")
        # store last variant for feedback weighting
        self._last_variant = variant
        self._last_text = dare

        update_word_freqs(dare, self.word_freqs)
        save_word_freq(self.word_freqs)

        return {"id": qid, "text": dare}

    def submit_keywords(self, round_number, question_id, keywords, players):
        # Persist keywords into engine state for this and future generations
        self.last_keywords = ", ".join(keywords) if isinstance(keywords, (list, tuple)) else str(keywords or "")
        # refresh ctx to include keywords
        self.ctx = ctx_from_players(players or self.players, self.ctx, self.last_keywords)
        print(f"[Engine] submit_keywords round={round_number} qid={question_id} keywords={self.last_keywords} players={players}")

    def mark_round_started(self, round_number, question_id, players, keywords=None):
        print(f"[Engine] mark_round_started round={round_number} qid={question_id} players={players} keywords={keywords}")

    def submit_feedback(self, round_number, question_id, rating, players, keywords=None):
        """
        Store feedback and update weights (bandit), mirroring app.py behavior.
        """
        txt = getattr(self, "_last_text", None)
        variant = getattr(self, "_last_variant", list(self.weights.keys())[0] if self.weights else "v4_spicy")
        if txt:
            append_feedback(variant, txt, int(rating) if rating is not None else 0, "")
            update_weights(self.weights, variant, int(rating) if rating is not None else 0)
            print(f"[Engine] submit_feedback round={round_number} qid={question_id} rating={rating} players={players} keywords={keywords}")
        else:
            print("[Engine] No last text to attach feedback to. Skipping.")

# ===============================
# === UI (from UI.py), wired ====
# ===============================

import tkinter as tk
from tkinter import ttk, messagebox

SAVE_FILE = "last_players.json"
LOADING_SECONDS = 10  # 10-second loading screen
MAX_PLAYERS = 10

# Use the real Engine defined above
ENGINE = Engine()

class App(tk.Tk):
    # --- palette (high-contrast, dark UI) ---
    APP_BG   = "#11161d"   # bg
    CARD_BG  = "#111827"
    FG       = "#F9FAFB"   # primary text
    FG_SOFT  = "#E5E7EB"
    FG_MUTED = "#9CA3AF"
    FG_HELP  = "#6B7280"
    ACCENT   = "#8B5CF6"   # purple
    ACCENT_A = "#7C3AED"
    OK       = "#22C55E"   # green
    OK_A     = "#16A34A"
    NEUTRAL  = "#A1AAB8"   # gray button
    NEUTRAL_A= "#4B5563"

    def __init__(self):
        super().__init__()
        self.title("What are the odds...? — Game")
        self.geometry("800x560")
        self.minsize(720, 520)

        # --- game state ---
        self.round_total = 0
        self.round_index = 0
        self.keyword_list: list[str] = []
        self.keyword_quota = 0
        self.players: list[str] = []
        self.rating = None
        self.started = False
        self.round = 1

        # --- engine / question state ---
        self.engine = ENGINE
        self.current_question_id = None
        self.current_question_text = (
            "Placeholder Question:\n\nWhat are the odds... "
            "(this will be replaced by your imported questions)?"
        )
        self.current_keywords = ""

        # --- container + screens ---
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        self.frames = {}
        for F in (StartScreen, PlayerEntryScreen, LoadGameScreen, LoadingScreen, QuestionScreen):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # --- theme / styling ---
        self._setup_theme()

        self.show("StartScreen")

    # ---------- UI helpers ----------
    def _setup_theme(self):
        style = ttk.Style(self)
        # try nice theme; fall back quietly
        try:
            self.tk.call("source", "sun-valley.tcl")
            style.theme_use("sun-valley-dark")
        except Exception:
            pass

        # root bg
        self.configure(bg=self.APP_BG)

        # base widgets on dark bg
        for s in ("TFrame", "TLabel"):
            style.configure(s, background=self.APP_BG)
        style.configure("Title.TLabel",    font=("Segoe UI", 18, "bold"), foreground=self.FG_SOFT)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12),         foreground=self.FG_SOFT)
        style.configure("Help.TLabel",     font=("Segoe UI", 10),         foreground=self.FG_HELP)
        style.configure("Question.TLabel", font=("Segoe UI", 16, "bold"), foreground=self.FG)

        # cards / panels
        style.configure("Card.TFrame", background=self.CARD_BG)

        # entries (light text on dark field)
        style.configure("TEntry",
                        fieldbackground=self.CARD_BG,
                        foreground=self.FG,
                        insertcolor=self.FG)

        # buttons: white text on colored backgrounds for contrast
        style.configure("Accent.TButton",   font=("Segoe UI", 11, "bold"))
        style.configure("Success.TButton",  font=("Segoe UI", 11, "bold"))
        style.configure("Secondary.TButton",font=("Segoe UI", 11, "bold"))

        style.map("Accent.TButton",
                  background=[("disabled", self.ACCENT_A), ("active", self.ACCENT_A), ("!disabled", self.ACCENT)],
                  foreground=[("disabled", self.FG_MUTED), ("!disabled", "#FFFFFF")])
        style.map("Success.TButton",
                  background=[("disabled", self.OK_A), ("active", self.OK_A), ("!disabled", self.OK)],
                  foreground=[("disabled", self.FG_MUTED), ("!disabled", "#FFFFFF")])
        style.map("Secondary.TButton",
                  background=[("disabled", self.NEUTRAL_A), ("active", self.NEUTRAL_A), ("!disabled", self.NEUTRAL)],
                  foreground=[("disabled", self.FG_MUTED), ("!disabled", "#FFFFFF")])

        # tk Listbox (used on Load screen) – force dark with light text
        self.option_add("*Listbox.Background",       self.NEUTRAL)
        self.option_add("*Listbox.Foreground",       self.FG)
        self.option_add("*Listbox.SelectBackground", self.NEUTRAL_A)
        self.option_add("*Listbox.SelectForeground", "#FFFFFF")
        self.option_add("*Listbox.Font",            ("Segoe UI", 11))

    def show(self, name: str):
        frame = self.frames[name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()


    def save_players(self):
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump({"players": self.players}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showwarning("Save failed", f"Couldn't save players:\n{e}")

    def load_players(self):
        if not os.path.exists(SAVE_FILE):
            return []
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("players", [])
        except Exception:
            return []
    
    def ask_keywords_for_round(self, total_hint: int | None = None):
        """Ask the user for round keywords and how many questions should include them."""
        if total_hint is None:
            total_hint = max(1, len(self.players) * 5)

        # 1) keywords (optional)
        ks = simpledialog.askstring(
            "Round keywords",
            "Enter keywords for this round (comma-separated, optional):",
            parent=self
        )
        if not ks:
            self.keyword_list = []
            self.keyword_quota = 0
            self.current_keywords = ""
            return

        kws = [k.strip() for k in ks.split(",") if k.strip()]
        self.keyword_list = kws
        self.current_keywords = ", ".join(kws)

        # 2) quota (how many questions should include >=1 keyword)
        q = simpledialog.askinteger(
            "Keyword quota",
            f"How many of the {total_hint} questions should include at least one of these keywords?\n(0–{total_hint})",
            parent=self,
            minvalue=0,
            maxvalue=total_hint,
        )
        if q is None:
            q = 0
        self.keyword_quota = int(q)




class StartScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        header = ttk.Label(self, text="What are the odds...?", style="Title.TLabel")
        header.pack(pady=(30, 10))

        sub = ttk.Label(
            self,
            text="A lightweight UI powered by your local generator.",
            style="Subtitle.TLabel",
        )
        sub.pack(pady=(0, 30))

        btns = ttk.Frame(self)
        btns.pack()

        def go_start():
            if self.controller.engine:
                try:
                    self.controller.engine.start_new_game()
                except Exception as e:
                    print(f"[UI] start_new_game error: {e}")
            self.controller.show("PlayerEntryScreen")

        start_btn = ttk.Button(btns, text="Start Game", command=go_start, style="Accent.TButton")
        load_btn  = ttk.Button(btns, text="Load Game",  command=lambda: controller.show("LoadGameScreen"), style="Secondary.TButton")

        start_btn.grid(row=0, column=0, padx=8, pady=8, ipadx=8, ipady=4)
        load_btn.grid(row=0, column=1, padx=8, pady=8, ipadx=8, ipady=4)

        footer = ttk.Label(self, text="© Roger and P to the K Production", style="Help.TLabel")
        footer.pack(side="bottom", pady=10)


class PlayerEntryScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller
        self.entries = []

        title = ttk.Label(self, text="Enter up to 10 player names", style="Title.TLabel")
        title.pack(pady=(24, 6))

        info = ttk.Label(self, text="Leave unused fields blank. You can add fewer than 10.", style="Subtitle.TLabel")
        info.pack(pady=(0, 18))

        form_wrap = ttk.Frame(self)
        form_wrap.pack(pady=8)

        grid = ttk.Frame(form_wrap)
        grid.pack()

        # Create 10 entry fields in two columns
        MAX_PLAYERS = 10
        for i in range(MAX_PLAYERS):
            row = i % 5
            col = i // 5
            lbl = ttk.Label(grid, text=f"Player {i+1}:")
            ent = ttk.Entry(grid, width=28)
            lbl.grid(row=row, column=col * 2, sticky="e", padx=(0, 6), pady=6)
            ent.grid(row=row, column=col * 2 + 1, sticky="w", padx=(0, 18), pady=6)
            self.entries.append(ent)

        btns = ttk.Frame(self)
        btns.pack(pady=16)

        self.start_btn = ttk.Button(btns, text="Continue ➜ Loading", command=self._continue)
        back_btn = ttk.Button(btns, text="Back", command=lambda: controller.show("StartScreen"))
        self.start_btn.grid(row=0, column=0, padx=6, ipadx=10, ipady=4)
        back_btn.grid(row=0, column=1, padx=6, ipadx=10, ipady=4)

    def on_show(self):
        for e in self.entries:
            e.delete(0, tk.END)
        self.entries[0].focus_set()

    def _continue(self):
        names = [e.get().strip() for e in self.entries]
        names = [n for n in names if n]
        if not names:
            messagebox.showinfo("Add players", "Please enter at least one player.")
            return
        if len(names) > 10:
            names = names[:10]
        self.controller.players = names
        self.controller.save_players()

        if self.controller.engine:
            try:
                self.controller.engine.save_players(self.controller.players)
            except Exception as e:
                print(f"[UI] save_players error: {e}")
        # Ask for this round's keywords & quota
        self.controller.ask_keywords_for_round(total_hint=max(1, len(self.controller.players) * 5))
        self.controller.started = False
        self.controller.rating = None
        self.controller.show("LoadingScreen")


class LoadGameScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        title = ttk.Label(self, text="Load Last Game", style="Title.TLabel")
        title.pack(pady=(24, 10))

        self.info = ttk.Label(self, text="", style="Subtitle.TLabel")
        self.info.pack(pady=(0, 12))

        self.list_frame = ttk.Frame(self)
        self.list_frame.pack(fill="both", expand=False, padx=12)

        self.listbox = tk.Listbox(self.list_frame, height=8, activestyle="dotbox")
        # match the dark background
        self.listbox.configure(
            bg="#1f2937",
            fg="#e5e7eb",
            highlightthickness=0,
            selectbackground="#374151",
            selectforeground="#ffffff",
            relief="flat",
        )
        sb = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)

        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.list_frame.columnconfigure(0, weight=1)
        self.list_frame.rowconfigure(0, weight=1)

        btns = ttk.Frame(self)
        btns.pack(pady=16)
        self.cont_btn = ttk.Button(btns, text="Continue ➜ Loading", command=self._continue)
        back_btn = ttk.Button(btns, text="Back", command=lambda: controller.show("StartScreen"))

        self.cont_btn.grid(row=0, column=0, padx=6, ipadx=10, ipady=4)
        back_btn.grid(row=0, column=1, padx=6, ipadx=10, ipady=4)

    def on_show(self):
        self.listbox.delete(0, tk.END)
        players = self.controller.load_players()
        if players:
            for p in players:
                self.listbox.insert(tk.END, p)
            self.info.config(text="Players from your last game:")
            self.cont_btn.state(["!disabled"])
        else:
            self.info.config(text="No previous game found.")
            self.cont_btn.state(["disabled"])

    def _continue(self):
        players = [self.listbox.get(i) for i in range(self.listbox.size())]
        if not players:
            messagebox.showinfo("Nothing to load", "There is no saved game to load.")
            return
        self.controller.players = players

        if self.controller.engine:
            try:
                self.controller.engine.save_players(self.controller.players)
            except Exception as e:
                print(f"[UI] save_players (load) error: {e}")
        # Ask for this round's keywords & quota
        self.controller.ask_keywords_for_round(total_hint=max(1, len(self.controller.players) * 5))
        self.controller.started = False
        self.controller.rating = None
        self.controller.show("LoadingScreen")


class LoadingScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        # Big title we animate: Loading. Loading.. Loading...
        self.title = ttk.Label(self, text="Loading", style="Title.TLabel")
        self.title.pack(pady=(36, 8))

        self.subtitle = ttk.Label(self, text="Preparing your game…", style="Subtitle.TLabel")
        self.subtitle.pack(pady=(0, 8))

        # Detail line for fetched counts / messages
        self.status = ttk.Label(self, text="", style="Subtitle.TLabel")
        self.status.pack(pady=(0, 6))

        # timers/state
        self._use_real = False
        self._dots = 0
        self._dots_job = None
        self._poll_job = None
        self._wait_ms = 0

    def on_show(self):
        # Reset UI & timers
        self._cancel_jobs()
        self._dots = 0
        self._wait_ms = 0
        self.title.config(text="Loading")
        self.subtitle.config(text="Preparing your game…")
        self.status.config(text="")
        self._tick_dots()

        # Try real progress (but we only *display* text, no bar)
        eng = self.controller.engine
        self._use_real = bool(eng and hasattr(eng, "begin_fetch_questions") and hasattr(eng, "get_fetch_status"))
        if self._use_real:
            try:
                total = max(1, len(self.controller.players) * 5)
                self.controller.round_total = total  # header in next screen uses this
                eng.begin_fetch_questions(
                    total=total,
                    kw_list=self.controller.keyword_list,
                    kw_quota=self.controller.keyword_quota,
                )
                self.subtitle.config(text=f"Fetching {total} questions…")
                self._poll_real()
            except Exception as e:
                self.status.config(text="(Falling back without progress)")
                self._fallback_wait()
        else:
            self._fallback_wait()

    def _tick_dots(self):
        # Animate title: Loading. Loading.. Loading...
        self._dots = (self._dots % 3) + 1
        self.title.config(text="Loading" + "." * self._dots)
        self._dots_job = self.after(400, self._tick_dots)

    def _poll_real(self):
        eng = self.controller.engine
        try:
            st = eng.get_fetch_status() if eng else None
        except Exception:
            st = None

        if isinstance(st, dict):
            loaded = st.get("loaded")
            total = st.get("total")
            done = bool(st.get("done"))
            error = st.get("error")
        else:
            loaded = total = None
            done = False
            error = None

        if error:
            self.subtitle.config(text="Error while fetching questions.")
            self.status.config(text=str(error))
            return self._fallback_wait()

        # Text-only feedback (no bar)
        if loaded is not None and total:
            self.status.config(text=f"Fetched {loaded}/{total}")
        else:
            self.status.config(text="Fetching…")

        if done:
            return self._proceed_to_question()

        self._poll_job = self.after(200, self._poll_real)

    def _fallback_wait(self):
        # Simple timed wait with the dot animation
        self._wait_ms += 200
        if self._wait_ms >= LOADING_SECONDS * 1000:
            self._wait_ms = 0
            return self._proceed_to_question()
        self._poll_job = self.after(200, self._fallback_wait)

    def _proceed_to_question(self):
        eng = self.controller.engine
        q = None
        if eng and hasattr(eng, "consume_prefetched_question"):
            try:
                q = eng.consume_prefetched_question()
            except Exception:
                q = None
        if q is None and eng:
            try:
                q = eng.get_next_question()
            except Exception:
                q = None

        if q:
            self.controller.current_question_id = q.get("id")
            self.controller.current_question_text = q.get("text")
            self.controller.frames["QuestionScreen"].q_lbl.config(
                text=self.controller.current_question_text
            )
        else:
            self.controller.current_question_id = None
            self.controller.current_question_text = None
            self.controller.frames["QuestionScreen"].q_lbl.config(
                text="No more questions available."
            )

        # Initialize round counters for next screen
        self.controller.round_total = max(1, len(self.controller.players) * 5)
        self.controller.round_index = 1
        self.controller.started = False

        self._cancel_jobs()
        self.controller.show("QuestionScreen")

    def _cancel_jobs(self):
        if self._dots_job is not None:
            try: self.after_cancel(self._dots_job)
            except Exception: pass
            self._dots_job = None
        if self._poll_job is not None:
            try: self.after_cancel(self._poll_job)
            except Exception: pass
            self._poll_job = None

    def destroy(self):
        self._cancel_jobs()
        super().destroy()



class Collapsible(ttk.Frame):
    """A small collapsible panel for the 'Rate Question' section."""
    def __init__(self, parent, title="Section"):
        super().__init__(parent)
        self._open = tk.BooleanVar(value=False)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        self.btn = ttk.Checkbutton(bar, text=title, variable=self._open, command=self._toggle, style="TCheckbutton")
        self.btn.pack(side="left", padx=2, pady=2)
        self.body = ttk.Frame(self)
        # initially collapsed

    def _toggle(self):
        if self._open.get():
            self.body.pack(fill="x", pady=(4, 0))
        else:
            self.body.forget()


class QuestionScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        header = ttk.Frame(self)
        header.pack(fill="x", pady=(24, 12), padx=16)

        self.title = ttk.Label(header, text="Round 1", style="Title.TLabel")
        self.title.pack(side="left")

        self.players_lbl = ttk.Label(header, text="", style="Subtitle.TLabel")
        self.players_lbl.pack(side="right")

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=16)

        q_card = ttk.Frame(content, padding=16, style="Card.TFrame")
        q_card.pack(fill="x", pady=(0, 12))
        self.q_lbl = ttk.Label(
            q_card,
            text="Placeholder Question:\n\nWhat are the odds... (this will be replaced by your imported questions)?",
            style="Question.TLabel",
            wraplength=720,
            justify="left",
        )
        self.q_lbl.pack(anchor="w")

        kw_row = ttk.Frame(content)
        kw_row.pack(fill="x", pady=(8, 6))
        ttk.Label(kw_row, text="Round keywords (comma-separated):").pack(side="left", padx=(0, 8))
        self.kw_entry = ttk.Entry(kw_row)
        self.kw_entry.pack(side="left", fill="x", expand=True)
        self.kw_hint = ttk.Label(
            content,
            text="These keywords are sent to the engine at round start.",
            style="Help.TLabel",
        )
        self.kw_hint.pack(anchor="w", pady=(0, 8))

        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=(8, 6))

        self.start_btn = ttk.Button(actions, text="Start", command=self._start_clicked, style="Accent.TButton")
        self.finish_btn = ttk.Button(actions, text="Finished", command=self._finished_clicked, style="Success.TButton")
        self.back_btn = ttk.Button(actions, text="Go back", command=self._go_back_clicked, style="Secondary.TButton")

        self.start_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        self.finish_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        self.back_btn.pack(side="left", padx=4, ipadx=10, ipady=4)

        self.finish_btn.forget()
        self.back_btn.forget()

        self.rate_panel = Collapsible(content, title="Rate Question")
        self.rate_panel.pack(fill="x", pady=(12, 0))

        rp = self.rate_panel.body
        self.rating_var = tk.StringVar(value="")
        rate_row = ttk.Frame(rp)
        rate_row.pack(pady=(2, 4))
        for val in [-2, -1, 0, 1, 2]:
            b = ttk.Radiobutton(
                rate_row,
                text=str(val),
                value=str(val),
                variable=self.rating_var,
                command=self._rated,
            )
            b.pack(side="left", padx=6, pady=4)

        help_text = "-2: report/block   -1: skip   0: meh   1: like   2: love"
        self.help_lbl = ttk.Label(rp, text=help_text, style="Help.TLabel")
        self.help_lbl.pack(anchor="w", pady=(2, 10))

        self.feedback = ttk.Label(rp, text="", style="Subtitle.TLabel")
        self.feedback.pack(anchor="w", pady=(0, 8))

        footer = ttk.Frame(self)
        footer.pack(fill="x", side="bottom", pady=8, padx=16)
        self.status = ttk.Label(footer, text="", style="Help.TLabel")
        self.status.pack(side="left")

        home_btn = ttk.Button(footer, text="⟵ Main Menu", command=lambda: controller.show("StartScreen"), style="Secondary.TButton")
        home_btn.pack(side="right")

    def _update_question_view(self):
        # Header: Round X : N players → M questions
        n_players = len(self.controller.players)
        self.title.config(
            text=f"Round {self.controller.round} : {n_players} players \u2192 {self.controller.round_total} questions"
        )
        # Question line prefix "Question i/M: ..."
        text = self.controller.current_question_text or ""
        i = max(1, int(self.controller.round_index or 1))
        m = max(1, int(self.controller.round_total or 1))
        self.q_lbl.config(text=f"Question {i}/{m}: {text}")


    def on_show(self):
        self._update_question_view()

        self.kw_entry.delete(0, tk.END)
        if self.controller.current_keywords:
            self.kw_entry.insert(0, self.controller.current_keywords)

        self.rating_var.set("" if self.controller.rating is None else str(self.controller.rating))
        self.feedback.config(text=self._rating_text(self.controller.rating) if self.controller.rating is not None else "")

        if self.controller.started:
            self._show_started_state()
        else:
            self._show_not_started_state()
        self.status.config(text="Question loaded.")

    def _show_not_started_state(self):
        if not self.start_btn.winfo_ismapped():
            self.start_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        if self.finish_btn.winfo_ismapped():
            self.finish_btn.forget()
        if self.back_btn.winfo_ismapped():
            self.back_btn.forget()
        self.kw_entry.state(["disabled"])

    def _show_started_state(self):
        if self.start_btn.winfo_ismapped():
            self.start_btn.forget()
        if not self.finish_btn.winfo_ismapped():
            self.finish_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        if not self.back_btn.winfo_ismapped():
            self.back_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        self.kw_entry.state(["disabled"])

    def _parse_keywords(self, text: str):
        return [k.strip() for k in text.split(",") if k.strip()]

    def _start_clicked(self):
        # Parse keywords from entry
        keywords = self._parse_keywords(self.controller.current_keywords or "")

        if self.controller.engine and self.controller.current_question_id is not None:
            try:
                self.controller.engine.submit_keywords(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    keywords=keywords,
                    players=self.controller.players,
                )
                self.controller.engine.mark_round_started(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    players=self.controller.players,
                    keywords=keywords,
                )
            except Exception as e:
                print(f"[UI] keywords/start export error: {e}")

        self.controller.started = True
        self.status.config(text="Round started. Play!")
        self._show_started_state()

    def _finished_clicked(self):
        # Submit feedback exactly once per question
        keywords = self._parse_keywords(self.controller.current_keywords or "")
        if self.controller.engine and self.controller.current_question_id is not None:
            try:
                self.controller.engine.submit_feedback(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    rating=self.controller.rating,
                    players=self.controller.players,
                    keywords=keywords,
                )
            except Exception as e:
                print(f"[UI] submit_feedback error: {e}")

        # Avoid double-click spam during transition
        self.finish_btn.state(["disabled"])
        self._next_question_in_round()
        self.finish_btn.state(["!disabled"])



    def _next_round(self):
        self.controller.round += 1
        self.controller.started = False
        self.controller.rating = None
        self.rating_var.set("")
        self.feedback.config(text="")
        self.status.config(text=f"Moving to Round {self.controller.round}...")
        self._show_not_started_state()
        self.title.config(text=f"Round {self.controller.round}")

        self.controller.current_keywords = ""
        self.kw_entry.state(["!disabled"])
        self.kw_entry.delete(0, tk.END)

        q = None
        if self.controller.engine:
            try:
                q = self.controller.engine.get_next_question()
            except Exception as e:
                print(f"[UI] get_next_question (next round) error: {e}")

        if q:
            self.controller.current_question_id = q.get("id")
            self.controller.current_question_text = q.get("text")
            self.q_lbl.config(text=self.controller.current_question_text)
        else:
            self.controller.current_question_id = None
            self.controller.current_question_text = None
            self.q_lbl.config(text="No more questions. Thanks for playing!")
            if self.start_btn.winfo_ismapped():
                self.start_btn.forget()
            if self.finish_btn.winfo_ismapped():
                self.finish_btn.forget()
            if self.back_btn.winfo_ismapped():
                self.back_btn.forget()

    def _next_question_in_round(self):
        # If there are more questions in this round, consume next prefetched one
        if self.controller.round_index < self.controller.round_total:
            self.controller.round_index += 1
            q = None
            eng = self.controller.engine
            if eng and hasattr(eng, "consume_prefetched_question"):
                try:
                    q = eng.consume_prefetched_question()
                except Exception as e:
                    print(f"[UI] consume_prefetched_question error: {e}")
            if q is None and eng:
                try:
                    q = eng.get_next_question()
                except Exception as e:
                    print(f"[UI] get_next_question error: {e}")

            if q:
                self.controller.current_question_id = q.get("id")
                self.controller.current_question_text = q.get("text")
            else:
                # If nothing came back, clamp the index and message
                self.controller.current_question_id = None
                self.controller.current_question_text = "No more questions available."

            # Reset per-question UI
            self.controller.rating = None
            self.rating_var.set("")
            self.feedback.config(text="")
            # Keep the round running (Start should stay hidden, Finished visible)
            self.status.config(text=f"Next question ({self.controller.round_index}/{self.controller.round_total}).")
            self._show_started_state()
            self._update_question_view()
            return

        # End of round
        proceed = messagebox.askyesno(
            "Round complete",
            f"Round {self.controller.round} complete.\nProceed to Round {self.controller.round + 1}?"
        )
        if proceed:
            self._begin_next_round()
        else:
            # Back to main menu (or you could just leave them on the last screen)
            self.controller.show("StartScreen")

    def _begin_next_round(self):
        # Prepare next round counters/state
        self.controller.round += 1
        self.controller.round_index = 0
        self.controller.round_total = 0
        self.controller.started = False
        self.controller.rating = None
        self.rating_var.set("")
        self.feedback.config(text="")
        self.controller.current_keywords = ""
        self.kw_entry.state(["!disabled"])
        self.kw_entry.delete(0, tk.END)
        # Ask for next round's keywords/quota (optional but handy)
        self.controller.ask_keywords_for_round(total_hint=max(1, len(self.controller.players) * 5))


        # Show loading to prefetch the next batch (players × 5) with real progress
        self.controller.show("LoadingScreen")


    def _go_back_clicked(self):
        self.controller.started = False
        self.status.config(text="Returned to pre-start state.")
        self._show_not_started_state()

    def _rated(self):
        val = int(self.rating_var.get())
        self.controller.rating = val
        self.feedback.config(text=self._rating_text(val))
        self.status.config(text=f"Rating saved: {val}")
        # (removed) do not send to engine here to avoid double feedback


    @staticmethod
    def _rating_text(val):
        if val is None:
            return ""
        mapping = {
            -2: "You rated this question: -2 (Report/Block)",
            -1: "You rated this question: -1 (Skip)",
            0: "You rated this question: 0 (Meh)",
            1: "You rated this question: 1 (Like)",
            2: "You rated this question: 2 (Love)",
        }
        return mapping.get(val, "")


if __name__ == "__main__":
    try:
        App().mainloop()
    except KeyboardInterrupt:
        pass
