Here’s a **starter README.md** you can drop into your repo. It explains what your script does, how to install, and how to use it:

---

````markdown
# 🎲 What Are The Odds? (Local Dare Generator)

This is a small **two-person coding project** that implements the party game *“What Are The Odds?”* with **AI-generated dares**, entirely **locally** using [Ollama](https://ollama.ai/).  
You can give feedback (ratings) on dares, and over time the script learns your group’s style by training a tiny reward model.

---

## ✨ Features
- Generates dares in different vibes: **chill, silly, chaotic**  
- Safety levels: **PG, PG-13, Party (spicy)**  
- Keeps track of already-seen dares to avoid repeats  
- Rate dares from `-2` (block) to `+2` (love)  
- Optional regex blocking for unwanted patterns  
- Local reward model (TF-IDF + Ridge) trained from your ratings  
- Export full packs of dares for vacations, parties, etc.  
- Works fully offline once Ollama + model are downloaded

---

## 🛠 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourname/odds-game.git
   cd odds-game
````

2. **Set up Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Requirements (already in `requirements.txt`)

3. **Install Ollama**
   Download and install from [ollama.ai](https://ollama.ai).
   Then pull the models you want to use:

   ```bash
   ollama pull mistral
   ollama pull llama3.2:3b
   ```

---

## 🚀 Usage

### Generate and rate one dare

```bash
python app.py play
```

You’ll see one dare, then rate it:

* `2 = love`
* `1 = like`
* `0 = meh`
* `-1 = skip`
* `-2 = block` (optionally enter a regex to ban)

### Show the top N dares (ranked by your feedback model)

```bash
python app.py play-top --top-n 10
```

### Train the reward model

After rating \~30 dares:

```bash
python app.py train-reward
```

### Make a pack of dares (e.g. for vacation)

```bash
python app.py make-pack --total 100 --out vacation_pack.md
```


---

## 📂 Data & Storage

The script automatically saves its data in the `data/` folder:

* `feedback.csv` → your ratings
* `weights.json` → variant bandit scores
* `seen.json` → remembers past dares
* `blocked.csv` → blocked regexes
* `reward.joblib` + `reward_vocab.json` → trained reward model

---

## ⚡ Tips for Speed

* Increase batch size per generation to reduce calls to Ollama
* Parallel generation across variants
* Cached reward model loading (already included in code)
* Disable `llm_judge` unless you want slower but smarter scoring

---

## 📜 License

MIT — feel free to hack, remix, and play.

---

## 🙋 FAQ

**Q: Do I need internet?**
A: Only to download Ollama + models the first time. After that it’s 100% local.

**Q: Can I run this on my phone?**
A: You’ll need a device that can run Ollama (Mac, Linux, or Windows with WSL). On vacation, a laptop is easiest. For phone-only, you’d need to host on a local machine and connect.

**Q: What’s the difference between PG / PG13 / Party?**

* **PG:** safe, family-friendly
* **PG13:** mild embarrassment / goofy
* **Party:** spicier, more daring (but never illegal / hateful)

---

🎉 Have fun, and may the odds be ever in your favor!
