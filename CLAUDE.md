# autoresearch — session start

This repo is an autonomous LLM-training research org. Two docs drive it:
- **`program.md`** — the **mission & config** (what to optimize, the metric, the knobs, the
  search axes). Editable by you and the human.
- **`ENGINE.md`** — the **fixed engine** (how the orchestrator + subagents actually run a
  session). **Never edit `ENGINE.md`.**

At the **start of every session**, do ONE of the following:

## A) If this is a fresh clone — run onboarding FIRST

**Check `program.md` for the marker `<!-- AUTORESEARCH:UNCONFIGURED -->`.** If it's there,
the repo hasn't been pointed at a research goal yet. Before doing anything else (even if the
user asked something else, briefly say you'll get them set up first), **run onboarding**:

1. **Ask what they want to research**, using the `AskUserQuestion` tool. Cover, in order
   (one question at a time, drilling in with follow-ups until it's unambiguous):
   - **The research target** — what gets optimized. Offer sensible options, e.g.: *general
     (lowest `val_bpb` by editing all of `train.py`)*, *a loss function that generalizes*,
     *the optimizer*, *the architecture*, *data efficiency* — or a custom goal they type.
   - **The success metric** — confirm it's `val_bpb` (lower = better), or what else, and
     whether there's a secondary signal to watch (e.g. an overfitting gap).
   - **Two time budgets (ask for both, separately):** (a) **per-experiment** minutes — how
     long *one* `train.py` run trains (default 5); and (b) **session** hours — how long the
     *whole* run lasts before the box auto-destroys (default 3, typically 2–3 h).
   - **Hardware** — GPU type (default `RTX_4090`), parallel GPUs (default 4), price cap
     (default $0.60/GPU/hr).
   - **Any constraints** — things to avoid, must-keep, or specific ideas to try first.
   Ask as many follow-ups as needed to get it *exactly* right — this is the one chance to
   nail the direction.

2. **Tailor the repo to their answers:**
   - Rewrite `program.md` §1 (what we're optimizing), §2 (metric) and §4 (search axes) to
     match the goal. Make the axes mutually exclusive for *this* target (e.g. a loss search
     partitions by the math done to cross-entropy, not by model component).
   - Set the §3 config defaults (minutes / hours / gpus / price / shards) to their choices.
   - **At setup you may edit `prepare.py` freely** (this is the one allowed window for it —
     e.g. shrink `TRAIN_TOKENS` for a data-constrained generalization study) as well as
     `train.py`'s baseline. Say what you changed and why. (Once research launches, `prepare.py`
     and the metric are frozen — only `train.py` changes during the loop, so no cheating.)
   - **Remove the `<!-- AUTORESEARCH:UNCONFIGURED -->` marker line** from `program.md`.

3. **Confirm** the tailored mission back to the human in 2–3 lines, then offer to kick off a
   session (`vast.py start …`). Don't start renting GPUs until they say go.

## B) If already configured — proceed normally

No marker → the mission is set. Read `program.md` (mission + config) and `ENGINE.md`
(engine), then do what the user asked. To start a research session, follow `ENGINE.md`:
bring the box up with `python vast.py start …`, then **spawn one subagent per GPU slot**
(you orchestrate — you do **not** run experiments yourself) and run the round loop.

---

**Always:** never edit `ENGINE.md`. `prepare.py`'s `evaluate_bpb` and the `forward`→logits
contract are ground truth. The full control plane is `python vast.py --help`.
