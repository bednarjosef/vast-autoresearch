# autoresearch — session start

This is a general **autonomous research swarm**: an orchestrator rents a multi-GPU box and
runs a pool of parallel subagents that mutate one **experiment artifact** to improve a
**measured objective**, compounding wins over a session. **It can research almost anything**
that fits a simple contract (below) — optimizing an ML model, an algorithm, a GPU kernel, a
solver, a prompt, a trading rule, a compression scheme, a config… **LLM pretraining is just
the default instantiation that ships in the repo** (`train.py` + `prepare.py`, objective
`val_bpb`); it's filler to be reshaped to whatever the user wants.

Two docs drive a session:
- **`program.md`** — the **mission & config** (what to optimize, the metric, the knobs, the
  axes). Editable by you and the human.
- **`ENGINE.md`** — the **fixed engine** (how the orchestrator + subagents run). **Never edit it.**

The roles (the LLM default in parentheses):
- **The experiment** (`train.py`) — the one artifact the swarm edits; running it prints a
  `OBJECTIVE: <number>` line.
- **The harness** (`prepare.py`) — the **frozen** task/data/evaluator that computes the
  objective honestly, so the score can't be gamed.
- **The objective** — a metric name + direction, set via `vast.py start --metric NAME --goal min|max`.
- **The axes** — disjoint families of edits to the experiment (so subagents never overlap).

At the **start of every session**, do ONE of the following:

## A) Fresh clone — run onboarding FIRST

**Check `program.md` for the marker `<!-- AUTORESEARCH:UNCONFIGURED -->`.** If present, the
repo hasn't been pointed at a goal yet. Before anything else (even if the user asked something
else, say you'll get them set up first), **run onboarding**:

1. **Ask what they want to research**, with `AskUserQuestion` (one question at a time, drilling
   in with follow-ups until it's unambiguous):
   - **The research target.** What gets optimized? It can be **anything** — keep the LLM
     default, or point it elsewhere (another ML task, an algorithm/heuristic, a kernel, a
     solver, a prompt, a strategy…). Offer a few concrete options plus "something else (type it)".
   - **The objective + direction.** What single number defines success, and is **lower or
     higher** better? (LLM default: `val_bpb`, lower.) Note any secondary signal to watch.
   - **How a run is scored.** What does one experiment *do*, and how is the number computed —
     so we can make it a **frozen, un-gameable evaluator**? (For LLM: train, then `evaluate_bpb`.)
   - **Two time budgets, separately:** per-experiment minutes (one run; default 5) and session
     hours (whole run before auto-destroy; default 3, ~2–3 h typical).
   - **Hardware:** GPU type (default `RTX_4090`), parallel GPUs (default 4), price cap
     ($0.60/GPU/hr). If the task doesn't need a GPU, the box still has CPUs — note it.
   - **Constraints / must-keeps / ideas to try first.**

2. **Tailor the repo to their answers.**
   - **If keeping the LLM default:** just rewrite `program.md` §1/§2/§4 for their angle, set §3
     config, optionally tune `prepare.py` (e.g. `TRAIN_TOKENS`) and the `train.py` baseline.
   - **If retargeting to another domain — the adaptation playbook:**
     1. **Rewrite `train.py` as the experiment** for the new task. It must: run one trial and
        **print exactly one `OBJECTIVE: <number>`** summary line (named whatever you choose),
        plus any diagnostics as extra `name: number` lines. Keep it self-contained and
        deterministic where possible.
     2. **Rewrite `prepare.py` as the frozen harness/evaluator** — one-time setup (data/assets)
        + the function that computes the objective. **Reuse its deadline helpers**
        (`start_training_clock` / `except TrainingTimeUp` / `stop_training_clock`) so every run
        is hard-bounded to the per-experiment budget and still emits a final score.
     3. **Set the objective:** the session will use `vast.py start --metric OBJECTIVE --goal min|max`.
        Record that name/direction in `program.md` §2/§3.
     4. **Define disjoint axes in `program.md` §4** for the new domain — partition by *what part
        of the experiment* a change touches, so two slots can never try the same thing.
     5. **Adjust deps** in `pyproject.toml` if the task needs different libraries (installed at setup).
   - In all cases set §3 config to their choices and **remove the `<!-- AUTORESEARCH:UNCONFIGURED -->`
     marker** from `program.md`.

3. **Smoke-test the contract before declaring done.** Once a box is up (or locally if it has the
   right compute), run the experiment once and confirm it prints the `OBJECTIVE:` line and that
   `vast.py exp` parses it. Fix until it runs clean — the whole point is that it works out of the box.

4. **Confirm** the tailored mission to the human in 2–3 lines, then offer to kick off a session
   (`vast.py start --metric … --goal … …`). Don't rent GPUs until they say go.

## B) Already configured — proceed normally

No marker → the mission is set. Read `program.md` + `ENGINE.md`, then do what the user asked.
To start a session, follow `ENGINE.md`: `python vast.py start --metric … --goal … …`, then
**spawn one subagent per GPU slot** (you orchestrate — you never run experiments yourself) and
run the round loop.

---

**Always:** never edit `ENGINE.md`. During research only the **experiment** file is edited; the
**harness/evaluator is frozen** so the objective can't be gamed. Full control plane:
`python vast.py --help`.
