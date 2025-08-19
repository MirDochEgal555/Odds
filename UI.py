import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

SAVE_FILE = "last_players.json"
LOADING_SECONDS = 10  # 10-second loading screen
MAX_PLAYERS = 10

# --------------------------------------------------------------------------------------
# OPTIONAL MOCK ENGINE (for demo/running standalone). Replace with your real program.
# Set USE_MOCK_ENGINE=False and plug in your own import/instance under the INTERFACE hook.
# --------------------------------------------------------------------------------------
USE_MOCK_ENGINE = True


class MockEngine:
    def __init__(self):
        self.questions = [
            {"id": "q1", "text": "What are the odds you can balance a spoon on your nose for 10 seconds?"},
            {"id": "q2", "text": "What are the odds you can name 10 countries in 20 seconds?"},
            {"id": "q3", "text": "What are the odds you can do 20 jumping jacks right now?"},
        ]
        self.index = 0

    def start_new_game(self):
        self.index = 0
        print("[MockEngine] New game started.")

    def save_players(self, players):
        print(f"[MockEngine] Players saved: {players}")

    def get_next_question(self):
        if self.index >= len(self.questions):
            return None
        q = self.questions[self.index]
        self.index += 1
        print(f"[MockEngine] get_next_question -> {q}")
        return q

    def submit_keywords(self, round_number, question_id, keywords, players):
        print(f"[MockEngine] submit_keywords round={round_number} qid={question_id} keywords={keywords} players={players}")

    def mark_round_started(self, round_number, question_id, players, keywords=None):
        print(f"[MockEngine] mark_round_started round={round_number} qid={question_id} players={players} keywords={keywords}")

    def submit_feedback(self, round_number, question_id, rating, players, keywords=None):
        print(f"[MockEngine] submit_feedback round={round_number} qid={question_id} rating={rating} players={players} keywords={keywords}")


# === INTERFACE HOOK: import your functionality program / engine here ===
# Example:
# from your_engine_module import Engine
# ENGINE = Engine()
ENGINE = MockEngine() if USE_MOCK_ENGINE else None
# --------------------------------------------------------------------------------------


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("What are the odds...? — Game")
        self.geometry("800x560")
        self.minsize(720, 520)

        # Shared state
        self.players = []
        self.rating = None
        self.started = False
        self.round = 1

        # === INTERFACE STATE: storage for backend integration ===
        self.engine = ENGINE  # set to your engine instance
        self.current_question_id = None
        self.current_question_text = None
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
            text="A lightweight UI scaffold. Add your real questions later.",
            style="Subtitle.TLabel",
        )
        sub.pack(pady=(0, 30))

        btns = ttk.Frame(self)
        btns.pack()

        def go_start():
            # === INTERFACE (optional): signal a new game session ===
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
        self.entries: list[ttk.Entry] = []

        title = ttk.Label(self, text="Enter up to 10 player names", style="Title.TLabel")
        title.pack(pady=(24, 6))

        info = ttk.Label(self, text="Leave unused fields blank. You can add fewer than 10.", style="Subtitle.TLabel")
        info.pack(pady=(0, 18))

        form_wrap = ttk.Frame(self)
        form_wrap.pack(pady=8)

        grid = ttk.Frame(form_wrap)
        grid.pack()

        # Create 10 entry fields in two columns
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
        # Clear entries each time
        for e in self.entries:
            e.delete(0, tk.END)
        self.entries[0].focus_set()

    def _continue(self):
        names = [e.get().strip() for e in self.entries]
        names = [n for n in names if n]  # drop blanks
        if not names:
            messagebox.showinfo("Add players", "Please enter at least one player.")
            return
        if len(names) > MAX_PLAYERS:
            names = names[:MAX_PLAYERS]
        self.controller.players = names
        self.controller.save_players()

        # === INTERFACE: export player names to backend ===
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

        # === INTERFACE: export loaded player names to backend ===
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

        self.subtitle = ttk.Label(self, text="Preparing your game. Please wait 10 seconds.", style="Subtitle.TLabel")
        self.subtitle.pack(pady=(0, 24))

        self.progress = ttk.Progressbar(self, orient="horizontal", length=480, mode="determinate", maximum=LOADING_SECONDS * 10)
        self.progress.pack(pady=8)

        self.status = ttk.Label(self, text="", style="Help.TLabel")
        self.status.pack(pady=6)

        self._ticks = 0
        self._job = None

    def on_show(self):
        self._ticks = 0
        self.progress["value"] = 0
        self.status.config(text="")
        # Update progress every 100 ms for LOADING_SECONDS seconds
        self._schedule_tick()

    def _schedule_tick(self):
        self._ticks += 1
        self.progress["value"] = self._ticks
        remaining = max(0, LOADING_SECONDS - self._ticks / 10)
        self.status.config(text=f"Time remaining: {remaining:0.1f}s")
        if self._ticks >= LOADING_SECONDS * 10:
            # === INTERFACE: fetch first question from backend ===
            if self.controller.engine:
                try:
                    q = self.controller.engine.get_next_question()
                except Exception as e:
                    print(f"[UI] get_next_question error: {e}")
                    q = None
            else:
                q = None

            if q:
                self.controller.current_question_id = q.get("id")
                self.controller.current_question_text = q.get("text")
                self.controller.frames["QuestionScreen"].q_lbl.config(text=self.controller.current_question_text)
            else:
                self.controller.current_question_id = None
                self.controller.current_question_text = None
                self.controller.frames["QuestionScreen"].q_lbl.config(text="No more questions available.")

            self.controller.show("QuestionScreen")
            return
        self._job = self.after(100, self._schedule_tick())

    def destroy(self):
        # Cancel pending after if any
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

        # Question text
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

        # Keywords row (before each round)
        kw_row = ttk.Frame(content)
        kw_row.pack(fill="x", pady=(8, 6))
        ttk.Label(kw_row, text="Round keywords (comma-separated):").pack(side="left", padx=(0, 8))
        self.kw_entry = ttk.Entry(kw_row)
        self.kw_entry.pack(side="left", fill="x", expand=True)
        self.kw_hint = ttk.Label(
            content,
            text="These keywords are sent to the functionality program at round start.",
            style="Help.TLabel",
        )
        self.kw_hint.pack(anchor="w", pady=(0, 8))

        # Action row: Start / Finished / Go back
        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=(8, 6))

        self.start_btn = ttk.Button(actions, text="Start", command=self._start_clicked)
        self.finish_btn = ttk.Button(actions, text="Finished", command=self._finished_clicked)
        self.back_btn = ttk.Button(actions, text="Go back", command=self._go_back_clicked)

        self.start_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        self.finish_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        self.back_btn.pack(side="left", padx=4, ipadx=10, ipady=4)

        # Initially only Start visible
        self.finish_btn.forget()
        self.back_btn.forget()

        # Collapsible: Rate Question
        self.rate_panel = Collapsible(content, title="Rate Question")
        self.rate_panel.pack(fill="x", pady=(12, 0))

        rp = self.rate_panel.body
        # Rating buttons row
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

        # Helper / explanation
        help_text = "-2: very bad   -1: bad   0: neutral   1: ok   2: very good"
        self.help_lbl = ttk.Label(rp, text=help_text, style="Help.TLabel")
        self.help_lbl.pack(anchor="w", pady=(2, 10))

        # Current rating feedback
        self.feedback = ttk.Label(rp, text="", style="Subtitle.TLabel")
        self.feedback.pack(anchor="w", pady=(0, 8))

        # Footer
        footer = ttk.Frame(self)
        footer.pack(fill="x", side="bottom", pady=8, padx=16)
        self.status = ttk.Label(footer, text="", style="Help.TLabel")
        self.status.pack(side="left")

        home_btn = ttk.Button(footer, text="⟵ Main Menu", command=lambda: controller.show("StartScreen"))
        home_btn.pack(side="right")

    def on_show(self):
        # Reset UI based on controller state
        self.title.config(text=f"Round {self.controller.round}")
        self.players_lbl.config(text=f"Players: {', '.join(self.controller.players)}")

        # === INTERFACE: if backend already provided a question, show it ===
        if self.controller.current_question_text:
            self.q_lbl.config(text=self.controller.current_question_text)

        # Reset keywords entry for the new round (keep controller state if needed)
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
        # Show only Start
        if not self.start_btn.winfo_ismapped():
            self.start_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        if self.finish_btn.winfo_ismapped():
            self.finish_btn.forget()
        if self.back_btn.winfo_ismapped():
            self.back_btn.forget()
        # Ensure keywords editable before round
        self.kw_entry.state(["!disabled"])

    def _show_started_state(self):
        # Show Finished + Go back, hide Start
        if self.start_btn.winfo_ismapped():
            self.start_btn.forget()
        if not self.finish_btn.winfo_ismapped():
            self.finish_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        if not self.back_btn.winfo_ismapped():
            self.back_btn.pack(side="left", padx=4, ipadx=10, ipady=4)
        # Lock keywords while round is running (optional)
        self.kw_entry.state(["disabled"])

    def _parse_keywords(self, text: str):
        # split by comma, strip whitespace, drop empties
        return [k.strip() for k in text.split(",") if k.strip()]

    def _start_clicked(self):
        # Read and export keywords BEFORE the round starts
        self.controller.current_keywords = self.kw_entry.get().strip()
        keywords = self._parse_keywords(self.controller.current_keywords)

        # === INTERFACE: send keywords + notify backend that round is starting ===
        if self.controller.engine and self.controller.current_question_id is not None:
            try:
                # export keywords (explicit call)
                self.controller.engine.submit_keywords(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    keywords=keywords,
                    players=self.controller.players,
                )
                # notify start (with keywords)
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
        # === INTERFACE: finalize/submit feedback for the finished round ===
        keywords = self._parse_keywords(self.controller.current_keywords)
        if self.controller.engine and self.controller.current_question_id is not None:
            try:
                self.controller.engine.submit_feedback(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    rating=self.controller.rating,  # may be None if not rated
                    players=self.controller.players,
                    keywords=keywords,
                )
            except Exception as e:
                print(f"[UI] submit_feedback error: {e}")

        # Advance to the next round no matter what
        self._next_round()

    def _next_round(self):
        # Increment round, reset per-round state, and refresh the screen
        self.controller.round += 1
        self.controller.started = False
        # Reset rating per round (optional but likely desired)
        self.controller.rating = None
        self.rating_var.set("")
        self.feedback.config(text="")
        self.status.config(text=f"Moving to Round {self.controller.round}...")
        self._show_not_started_state()
        self.title.config(text=f"Round {self.controller.round}")

        # Clear keywords field for next round
        self.controller.current_keywords = ""
        self.kw_entry.state(["!disabled"])
        self.kw_entry.delete(0, tk.END)

        # === INTERFACE: get next question from backend ===
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
            # Optionally disable Start/Finished when no questions remain
            if self.start_btn.winfo_ismapped():
                self.start_btn.forget()
            if self.finish_btn.winfo_ismapped():
                self.finish_btn.forget()
            if self.back_btn.winfo_ismapped():
                self.back_btn.forget()

    def _go_back_clicked(self):
        self.controller.started = False
        self.status.config(text="Returned to pre-start state.")
        self._show_not_started_state()

    def _rated(self):
        val = int(self.rating_var.get())
        self.controller.rating = val
        self.feedback.config(text=self._rating_text(val))
        self.status.config(text=f"Rating saved: {val}")

        # === INTERFACE: export rating feedback immediately (optional immediate push) ===
        if self.controller.engine and self.controller.current_question_id is not None:
            keywords = self._parse_keywords(self.controller.current_keywords)
            try:
                self.controller.engine.submit_feedback(
                    round_number=self.controller.round,
                    question_id=self.controller.current_question_id,
                    rating=val,
                    players=self.controller.players,
                    keywords=keywords,
                )
            except Exception as e:
                print(f"[UI] submit_feedback (immediate) error: {e}")

    @staticmethod
    def _rating_text(val):
        if val is None:
            return ""
        mapping = {
            -2: "You rated this question: Very bad (-2)",
            -1: "You rated this question: Bad (-1)",
            0: "You rated this question: Neutral (0)",
            1: "You rated this question: OK (1)",
            2: "You rated this question: Very good (2)",
        }
        return mapping.get(val, "")


if __name__ == "__main__":
    App().mainloop()
