# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Spin up Claude Code in this repo (disable permissions for autonomy). **On a fresh clone the
agent onboards you automatically** — `CLAUDE.md` detects the unconfigured marker and asks
what you want to research (target, metric, per-experiment budget, GPUs, session length),
then tailors `program.md` to your goal. After that, just say "kick off a session".

The docs split into two roles:
- **`program.md`** — the **mission & config** you edit (what to optimize, the metric, the
  knobs, the search axes).
- **`ENGINE.md`** — the **fixed engine** (how the orchestrator + subagents run a session).
  Not edited by the human or the agent.

## Running on Vast.ai (this fork)

This fork adds **`vast.py`** so you can run the whole thing from a laptop with **no
local GPU**. The agents and the research loop stay on your machine; only the GPU work
(`train.py`, the per-experiment training budget) runs on **one rented Vast.ai box per session** with
several GPUs, so a pool of subagents can run experiments **in parallel** — one per GPU,
each pinned to its own GPU + CPU cores so they never interfere — while sharing a single
ledger (`results.tsv`) and knowledge log (`findings.md`).

**One-time:** `vastai set api-key <KEY>` (an ssh key is auto-created/registered).

**Each session,** the agent brings the box up with one command, then runs the loop:

```bash
# one-shot: rent cheapest qualifying box + background watchdog + setup + ~1-min bench
python vast.py start --gpus 4 --hours 3 --minutes 5 --max-price 0.60
python vast.py dashboard          # live browser view of progress (see below)
python vast.py reap               # between rounds: kill stray/ghost runs; show GPU procs
python vast.py down               # destroy the box + clear state
```

`--minutes` sets the per-experiment training budget (variable per session; same for every
run so they stay comparable). `start` rolls up the old `up`/`watchdog`/`setup`/`bench` steps
(they still exist individually) and warms each slot's compile cache so the first experiments
start fast.

**Orchestration (`ENGINE.md`).** The main agent is an **orchestrator**: each round it gives
every GPU slot a **disjoint axis** (e.g. optimizer / attention / MLP+norm / capacity /
embeddings+init) with concrete ideas and an explicit off-limits list, so subagents never
overlap. It picks **1 or 2 experiments per round** (same for all slots) and **spawns one
subagent per GPU — all foreground, in a single concurrent batch** (never backgrounded,
never polled; it never runs experiments itself). Each subagent runs exactly that many,
**then returns**. The orchestrator then **compounds**: it verifies wins, and when two land
it **tests their combination and adopts the best as the new base**, so the champion is
always "everything that's worked so far" and `val_bpb` keeps dropping over the whole session
(it builds *upon* the champion, not from scratch). **Anti-cheat:** during research only
`train.py` changes — `exp` uploads just that file, so the frozen `prepare.py`/metric score
every run. Runs are bounded by `train.py`'s own budget (no force-kill), the baseline fits a
24 GB GPU, OOMs are failures, and `reap` clears ghosts between rounds.

**Watch it live.** `python vast.py dashboard` starts a tiny local web server (stdlib
only, no deps) and opens a browser page that auto-refreshes every 5s: a **val_bpb chart**
over experiments (with a "best so far" line), a leaderboard, recent runs, and a live box
panel (GPUs, $/hr, uptime, **spend so far**, and **deadline countdown**). It just reads
the local `results.tsv` and `.vast_state.json`, so it updates as the swarm logs results
— leave it open overnight. For a quick one-shot status in the terminal, use
`python vast.py status`.

Safety is built in: exactly one tracked box at a time with a hard auto-destroy
deadline, a background `watchdog`, and `ps`/`nuke` to catch orphans. Run
`python vast.py --help` for all commands. At ~$0.34/GPU/hr for 4090s, a 3-hour 4-GPU
session is roughly $4.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
CLAUDE.md       — session-start router: onboards a fresh clone, else proceeds
program.md      — MISSION & CONFIG: what to optimize + knobs (edit this)
ENGINE.md       — fixed orchestration engine: how the swarm runs (do not edit)
vast.py         — Vast.ai control plane (start/exp/bench/reap/dashboard/teardown)
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
