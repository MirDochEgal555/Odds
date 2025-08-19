
#!/usr/bin/env python3
# merged_game.py
# Combines the generation/learning engine (from app.py) with the Tk UI (from UI.py)
# to provide a single runnable program.

import os, json, re, csv, random, pathlib, textwrap, argparse, urllib.request, sys, math, time
from datetime import datetime
import threading

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
Return a JSON array of exactly 30 distinct dares starting with "[name] What are the odds you ..." (strings only)."""

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
    """Combine reward model, heuristic, and optional LLM judge."""
    if not cands: return None
    r_preds = reward_predict(cands)
    have_reward = any(p is not None for p in r_preds)
    h_scores = [heuristic_score(t, ctx["vibe"], ctx["names_list"]) for t in cands]
    try:
        llm_scores = [llm_judge_score(t) for t in cands]
    except Exception:
        llm_scores = [0.5]*len(cands)
    fused = []
    for i,t in enumerate(cands):
        rp = r_preds[i] if have_reward else 0.5
        hs = h_scores[i]
        ls = llm_scores[i]
        score = (0.55*rp + 0.25*ls + 0.20*(hs/2.0))
        fused.append((score, t))
    fused.sort(reverse=True, key=lambda x:x[0])
    return fused[0][1]

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
    
    def begin_fetch_questions(self, total: int = 1):
        """Start background prefetch of `total` questions; non-blocking."""
        with self._fetch_lock:
            if self._fetch_thread is not None and self._fetch_thread.is_alive():
                return  # already running
            total = max(1, int(total or 1))
            self._fetch_status = {"loaded": 0, "total": total, "percent": 0.0,
                                  "message": "Starting…", "done": False, "error": None}

        def _worker():
            try:
                for _ in range(total):
                    q = self.get_next_question()
                    with self._fetch_lock:
                        if q:
                            self._prefetched.append(q)
                            self._fetch_status["loaded"] += 1
                            t = self._fetch_status["total"] or total
                            self._fetch_status["percent"] = (self._fetch_status["loaded"] * 100.0) / t
                            self._fetch_status["message"] = f"Fetched {self._fetch_status['loaded']}/{t}"
                        else:
                            self._fetch_status["message"] = "No question returned."
                            break
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
        """Pop one prefetched question if available."""
        with self._fetch_lock:
            if self._prefetched:
                return self._prefetched.pop(0)
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
    def __init__(self):
        super().__init__()
        self.title("What are the odds...? — Game")
        self.geometry("800x560")
        self.minsize(720, 520)
        self.round_total = 0     # total questions this round
        self.round_index = 0     # 1-based index within the round


        # Shared state
        self.players = []
        self.rating = None
        self.started = False
        self.round = 1

        # === INTERFACE STATE ===
        self.engine = ENGINE
        self.current_question_id = None
        this_text = "Placeholder Question:\n\nWhat are the odds... (this will be replaced by your imported questions)?"
        self.current_question_text = this_text
        self.current_keywords = ""  # raw comma-separated string for the round

        # Container for screens
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartScreen, PlayerEntryScreen, LoadGameScreen, LoadingScreen, QuestionScreen):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show("StartScreen")

        # Basic ttk styling
        style = ttk.Style(self)
        try:
            self.tk.call("source", "sun-valley.tcl")  # if present, use a nicer theme
            style.theme_use("sun-valley-dark")
        except Exception:
            style.theme_use(style.theme_use())
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12))
        style.configure("Help.TLabel", font=("Segoe UI", 10), foreground="#555")
        style.configure("Question.TLabel", font=("Segoe UI", 16, "bold"))

    def show(self, name):
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

        start_btn = ttk.Button(btns, text="Start Game", command=go_start)
        load_btn = ttk.Button(btns, text="Load Game", command=lambda: controller.show("LoadGameScreen"))

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

        self.controller.started = False
        self.controller.rating = None
        self.controller.show("LoadingScreen")


class LoadingScreen(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent)
        self.controller = controller

        title = ttk.Label(self, text="Loading...", style="Title.TLabel")
        title.pack(pady=(36, 12))

        self.subtitle = ttk.Label(self, text="Preparing your game. Please wait...", style="Subtitle.TLabel")
        self.subtitle.pack(pady=(0, 24))

        # We'll switch modes between determinate/indeterminate
        self.progress = ttk.Progressbar(self, orient="horizontal", length=480, mode="determinate", maximum=100)
        self.progress.pack(pady=8)

        self.status = ttk.Label(self, text="", style="Help.TLabel")
        self.status.pack(pady=6)

        self._job = None
        self._ticks = 0
        self._use_real = False
        self._fallback_msgs = [
            "Working on it…",
            "Optimizing your questions…",
            "Downloading question packs…",
            "Indexing categories…",
            "Almost there…",
        ]
        self._fallback_idx = 0

    def on_show(self):
        # Reset UI
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0
        self.status.config(text="")
        self._ticks = 0
        self._fallback_idx = 0

        # Try real progress if engine supports it
        eng = self.controller.engine
        self._use_real = bool(eng and hasattr(eng, "begin_fetch_questions") and hasattr(eng, "get_fetch_status"))
        if self._use_real:
            try:
                total = max(1, len(self.controller.players) * 5)
                eng.begin_fetch_questions(total=total)
                self.subtitle.config(text=f"Fetching {total} questions…")
                self.subtitle.config(text="Fetching questions…")
                self._poll_real()
            except Exception as e:
                print(f"[UI] begin_fetch_questions error: {e}")
                self._start_spinner()
        else:
            self._start_spinner()

    # -------- REAL PROGRESS PATH --------
    def _poll_real(self):
        eng = self.controller.engine
        try:
            st = eng.get_fetch_status() if eng else None
        except Exception as e:
            print(f"[UI] get_fetch_status error: {e}")
            st = None

        if isinstance(st, dict):
            percent = st.get("percent")
            loaded = st.get("loaded")
            total = st.get("total")
            message = st.get("message")
            done = bool(st.get("done"))
            error = st.get("error")
        else:
            percent = None; loaded = None; total = None; message = None; done = False; error = None

        if error:
            self.subtitle.config(text="Error while fetching questions.")
            self.status.config(text=str(error))
            # fallback so user isn't stuck
            return self._start_spinner()

        # Update progressbar
        if percent is not None:
            self.progress.configure(mode="determinate")
            try:
                self.progress["value"] = max(0, min(100, float(percent)))
            except Exception:
                self.progress["value"] = 0
        else:
            if str(self.progress.cget("mode")) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(100)

        # Update status line
        if message:
            if loaded is not None and total:
                self.status.config(text=f"{message}  ({loaded}/{total})")
            elif percent is not None:
                self.status.config(text=f"{message}  ({percent:.0f}%)")
            else:
                self.status.config(text=message)
        else:
            if loaded is not None and total:
                self.status.config(text=f"Fetching questions… ({loaded}/{total})")
            elif percent is not None:
                self.status.config(text=f"Fetching questions… {percent:.0f}%")
            else:
                self.status.config(text="Fetching questions…")

        if done:
            self.progress.stop()
            self.progress.configure(mode="determinate")
            self.progress["value"] = 100
            self.subtitle.config(text="Done. Preparing first question…")
            return self._proceed_to_question()
        # keep polling
        self._job = self.after(150, self._poll_real)

    # -------- FALLBACK SPINNER PATH --------
    def _start_spinner(self):
        self.subtitle.config(text="Working on updates…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(100)
        self._ticks = 0
        self._spin_step()

    def _spin_step(self):
        self._ticks += 1
        # rotate helper text every ~1.2s
        if self._ticks % 12 == 0:
            self._fallback_idx = (self._fallback_idx + 1) % len(self._fallback_msgs)
            self.status.config(text=self._fallback_msgs[self._fallback_idx])
        # after LOADING_SECONDS seconds, proceed
        if self._ticks >= LOADING_SECONDS * 10:
            self.progress.stop()
            return self._proceed_to_question()
        self._job = self.after(100, self._spin_step)

    # -------- proceed to question --------
    def _proceed_to_question(self):
        q = None
        eng = self.controller.engine
        if eng and hasattr(eng, "consume_prefetched_question"):
            try:
                q = eng.consume_prefetched_question()
            except Exception as e:
                print(f"[UI] consume_prefetched_question error: {e}")
                q = None
        if q is None and eng:
            try:
                q = eng.get_next_question()
            except Exception as e:
                print(f"[UI] get_next_question error: {e}")
                q = None

        if q:
            self.controller.current_question_id = q.get("id")
            self.controller.current_question_text = q.get("text")
            self.controller.frames["QuestionScreen"].q_lbl.config(text=self.controller.current_question_text)
        else:
            self.controller.current_question_id = None
            self.controller.current_question_text = None
            self.controller.frames["QuestionScreen"].q_lbl.config(text="No more questions available.")

            # set counters for Round 1
        self.controller.round_total = max(1, len(self.controller.players) * 5)
        self.controller.round_index = 1
        self.controller.started = False  # round hasn't started until user presses Start
        self.controller.show("QuestionScreen")

    def destroy(self):
        if self._job is not None:
            self.after_cancel(self._job)
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

        q_card = ttk.Frame(content, padding=16, relief="groove")
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

        self.start_btn = ttk.Button(actions, text="Start", command=self._start_clicked)
        self.finish_btn = ttk.Button(actions, text="Finished", command=self._finished_clicked)
        self.back_btn = ttk.Button(actions, text="Go back", command=self._go_back_clicked)

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

        home_btn = ttk.Button(footer, text="⟵ Main Menu", command=lambda: controller.show("StartScreen"))
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
        self.kw_entry.state(["!disabled"])

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
        self.controller.current_keywords = self.kw_entry.get().strip()
        keywords = self._parse_keywords(self.controller.current_keywords)

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
        keywords = self._parse_keywords(self.controller.current_keywords)
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
        self.finish_btn.state(["disabled"])
        self._next_question_in_round()
        self.finish_btn.state(["!disabled"])
        # Advance within round, or end-of-round prompt
        self._next_question_in_round()


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
