# autoresearch

An autonomous LLM-training research swarm you drive from a laptop with **no local GPU**.
You tell it what to optimize; it rents a multi-GPU box on [Vast.ai](https://vast.ai), spawns
a pool of AI subagents that edit `train.py` in parallel — each on its own GPU — and compounds
their wins into a steadily-improving model. Findings are grounded in the literature via the
`research` skill, progress is visible in a live dashboard, and the box tears itself down at a
hard deadline so you can leave it running overnight.

The training core is a single-file GPT (Muon + AdamW), forked from
[nanochat](https://github.com/karpathy/nanochat) / [autoresearch](https://github.com/karpathy/nanochat).

---

## How it works

- **You** pick the goal (lowest `val_bpb`, a better optimizer, a loss that generalizes, …),
  the per-experiment training budget, and how long the session runs.
- **The orchestrator** (the main agent) rents one Vast box with N GPUs and, each round, hands
  every GPU slot a **disjoint research direction** so no two subagents try the same thing.
- **N subagents** run in parallel — one per GPU — each editing `train.py`, running it, and
  reporting back. Every experiment is `train.py` trained for a fixed budget; the score is
  **`val_bpb`** (held-out bits-per-byte, lower = better, vocab-independent so changes compare
  fairly).
- **Wins compound.** The orchestrator verifies improvements, **stacks** them (and tests
  combinations), and makes the result the new champion — every round builds *upon* the best so
  far instead of starting from scratch. `val_bpb` keeps dropping across the whole session.

The metric (`evaluate_bpb`) and the data regime (`prepare.py`) are **frozen during research**
so the score can't be gamed — gains must come from `train.py` alone.

## The files

| File | Role |
|------|------|
| **`CLAUDE.md`** | Session-start router. On a fresh clone it **onboards you** (asks what to research) and tailors `program.md`. |
| **`program.md`** | The **mission & config** you edit: what to optimize, the metric, the knobs, the search axes. |
| **`ENGINE.md`** | The **fixed engine**: how the orchestrator + subagents run a session. Not edited. |
| **`train.py`** | The research target — the only file edited *during* research. |
| **`prepare.py`** | Data, tokenizer, regime, and the `evaluate_bpb` metric. Frozen during research. |
| **`vast.py`** | The control plane (rent / setup / bench / run / dashboard / teardown). |

## Quick start

```bash
# 1. One-time: authenticate Vast (an ssh key is auto-created/registered).
vastai set api-key <YOUR_KEY>

# 2. Open Claude Code in this repo (grant permissions for autonomy).
#    A fresh clone auto-onboards: it asks what you want to research, the per-experiment
#    minutes, the session hours, and the hardware — then tailors program.md to your goal.

# 3. Say "kick off a session". The agent brings a box up in one command:
python vast.py start --gpus 4 --hours 3 --minutes 5 --max-price 0.60
#    start = rent cheapest qualifying box → background watchdog → setup → ~1-min bench.
```

Then the swarm runs the loop on its own until the deadline (or you stop it). Two independent
time knobs: **`--minutes`** = how long *one* experiment trains; **`--hours`** = how long the
*whole* session runs before the box auto-destroys (typically 2–3 h).

> Already have a GPU and just want to smoke-test the trainer locally?
> `uv sync && uv run prepare.py && uv run train.py`.

## Research-driven ideas — the `research` skill

The swarm doesn't only brainstorm from memory. Every round the orchestrator leans on the
**`research` skill** (scholarly + web search) to:

- surface **novel ideas** and **current SOTA** for whatever is being optimized,
- check whether an idea is already known to work (or to fail), before spending a GPU on it,
- ground each slot's assigned direction in the literature rather than guesswork.

To keep the GPUs busy while it reads, a **"research scout" subagent** joins the same
foreground batch as the experiment subagents — so literature mining happens *concurrently*
with training, and fresh ideas arrive with the fresh numbers. `ENGINE.md` explicitly directs
the agent to use the skill for ideation, SOTA-hunting, and verification — not to rely on what
it already knows.

## Control plane (`vast.py`)

`python vast.py --help` for everything. The ones you'll see most:

| Command | What it does |
|---------|--------------|
| `start` | One-shot bring-up: rent → watchdog → setup → bench. |
| `exp --slot N --train F` | Run one experiment on GPU `N` (subagents call this; foreground/blocking). |
| `bench` | ~1-min objectivity check: confirms GPUs are equivalent, measures contention. |
| `dashboard` | Live browser view (chart, leaderboard, cost, deadline). |
| `reap` | Kill stray/ghost runs; show what's on each GPU. |
| `status` / `ps` | Tracked-box status / all account instances. |
| `down` / `nuke` | Destroy the box / destroy all boxes. |

## Watch it live

`python vast.py dashboard` starts a tiny local web server (stdlib only, no deps) and opens a
page that auto-refreshes every 5s: a **`val_bpb` chart** with a best-so-far line, a
leaderboard, recent runs, and a live box panel (GPUs, $/hr, uptime, **spend so far**,
**deadline countdown**). It reads the local `results.tsv` + `.vast_state.json`, so it updates
as the swarm logs results — leave it open overnight.

## Safety & cost

- **One tracked box at a time** with a **hard auto-destroy deadline** and a background
  `watchdog`; `ps`/`nuke` catch orphans, `reap` clears stray runs.
- Experiments are pinned to disjoint GPUs **and** CPU cores, so parallel runs don't interfere;
  `bench` quantifies any residual contention.
- At ~$0.34/GPU/hr for RTX 4090s, a 3-hour 4-GPU session is roughly **$4**.

## Credits & license

Forked from Andrej Karpathy's autoresearch; training core adapted from
[nanochat](https://github.com/karpathy/nanochat). MIT.
