<!-- AUTORESEARCH:UNCONFIGURED -->
<!-- ^ While this marker is present, the repo is a fresh clone that hasn't been pointed at
     a research goal yet. On session start, CLAUDE.md runs onboarding: it asks what you want
     to research, tailors the sections below, and removes this marker. To re-run onboarding
     later, just put the marker back. -->

# Mission & config — EDIT THIS FILE

This is the **editable** part of autoresearch: **what** to research, **how** to score it,
and the **knobs**. The fixed machinery (how the orchestrator actually runs the loop) is in
[`ENGINE.md`](ENGINE.md) — **do not edit that one.** When the human starts a session, read
this file, then run the loop in `ENGINE.md`.

> **autoresearch can optimize almost anything.** The sections below describe the **default**
> instantiation — LLM pretraining, where the *experiment* is `train.py`, the *harness* is
> `prepare.py`, and the *objective* is `val_bpb`. To research something else, onboarding
> (`CLAUDE.md`) reshapes those: the experiment is whatever artifact you mutate (it just prints
> `OBJECTIVE: <number>`), the harness is a frozen evaluator that computes that number, and the
> objective + direction are set via `vast.py start --metric NAME --goal min|max`. Everything in
> `ENGINE.md` (parallel slots, disjoint axes, compounding, safety) is the same for any domain.

---

## 1. What we're optimizing  ← edit this

**Goal:** find the `train.py` that reaches the **lowest held-out `val_bpb`** within the
fixed per-experiment training budget. Everything in `train.py` is fair game — architecture,
optimizer, hyperparameters, training loop, batch size, model size.

**Bias toward big wins.** Prefer **bold, novel, even architectural** changes with large upside
over tiny hyperparameter tweaks. Micro-tuning and ±0.001 gains are secondary; the search should
keep swinging for changes that could move the metric *a lot*. (See "SWING FOR BIG WINS" in `ENGINE.md`.)

> Onboarding rewrites this section for your actual goal (e.g. "discover a loss that
> generalizes", "find the best optimizer", "maximize data efficiency"). Keep it to a
> sentence or two: the object you're changing and what counts as success.

## 2. The metric  ← edit if your goal changes the score

- **`val_bpb`** — bits-per-byte on a **held-out** shard. **Lower is better.** Vocab-size
  independent, so architecture changes compare fairly. This is the score.
- Every run also prints `peak_vram_mb` — watch it (see VRAM in `ENGINE.md`).
- The metric is computed by `evaluate_bpb` in `prepare.py` and is **ground truth** — never
  change it; `train.py`'s `forward` must keep returning raw next-token logits.

## 3. Config knobs  ← edit these defaults to taste

The orchestrator passes these to `vast.py start`. Defaults:

There are **two independent time knobs** — pick both:
- **`--minutes`** — how long **each single experiment** trains (one `train.py` run).
- **`--hours`** — how long the **whole session** runs before the box auto-destroys (typically 2–3 h).

| Knob | Default | What it controls |
|------|---------|------------------|
| **Objective** | `--metric val_bpb` | the metric name the experiment prints (`OBJECTIVE: <number>`). |
| **Direction** | `--goal min` | optimize it to `min` (lower better) or `max` (higher better). |
| **Per-experiment budget** | `--minutes 5` | how long **one** experiment runs. Same for every run in the session (so they stay comparable). Shorter = more experiments/hr; longer = more signal per run. |
| **Session length** | `--hours 3` | how long the whole research run lasts — the box's hard auto-destroy deadline (typically 2–3 h). |
| **GPUs / parallel slots** | `--gpus 4` | one experiment runs per GPU; this is the max parallelism. |
| **Price cap** | `--max-price 0.60` | $/GPU/hr ceiling. |
| **GPU type** | `--gpu RTX_4090` | which GPU to rent. |
| **Data shards** | `--num-shards 8` | how much training data to download at setup (more = slower setup; LLM-default only). |

One-shot bring-up: `python vast.py start --metric val_bpb --goal min --gpus 4 --hours 3 --minutes 5 --max-price 0.60`.

## 4. The search axes  ← edit for your research focus

The orchestrator gives each slot a **disjoint axis** each round so slots never overlap
(see `ENGINE.md`). Partition by **which component of `train.py` a change touches** — each
idea belongs to exactly one axis, so distinct axes can't collide:

- **Axis A — Optimizer & LR schedule.** Muon/AdamW hyperparams (lr, betas, eps, weight
  decay, momentum), warmup/decay shape, LR-vs-width scaling, grad clipping.
- **Axis B — Attention.** `n_head`, `n_kv_head` (GQA), `window_pattern` (local/global mix),
  rotary base/θ, attention scale / logit softcap, QK-norm.
- **Axis C — MLP, activation & normalization.** MLP ratio / hidden dim, activation function,
  norm type & placement (pre/post, RMS), residual scaling.
- **Axis D — Capacity & token budget.** `DEPTH`/`n_layer`, `n_embd`, `DEVICE_BATCH_SIZE` /
  `TOTAL_BATCH_SIZE` / grad-accum, precision/dtype, packing. **Owns all batch/size changes —
  the VRAM-sensitive axis.**
- **Axis E — Embeddings, init & regularization.** Value embeddings, weight tying,
  embedding/init scale, logit/z-loss, dropout, init schemes.

> Onboarding tailors these to your goal (e.g. a loss search partitions by the *math* the loss
> does to cross-entropy, not by model component). Keep them mutually exclusive — that's what
> makes non-overlap airtight.

## 5. What's editable — and when (no cheating)

The boundary is **temporal**, and it's what keeps the score honest:

- **Before research (setup / onboarding):** the human and their agent may edit anything to
  configure the study — `program.md`, `train.py`, and even **`prepare.py`** (the regime,
  data, tokenizer, `TRAIN_TOKENS`, …). Set it up however you like, then launch.
- **During research (the round loop):** **only `train.py` may change.** The orchestrator
  must NOT touch `prepare.py`, `evaluate_bpb`, the `forward`→logits contract,
  or [`ENGINE.md`](ENGINE.md). Every gain must come from `train.py` alone, so the metric
  can't be gamed. This is enforced structurally: `vast.py exp`/`round` upload **only** each
  slot's `train.py`, so the box keeps the `prepare.py`/metric frozen at setup no matter what —
  editing `prepare.py` locally simply has no effect on the score.
- **Keep the time-budget + eval scaffolding in `train.py`** when editing it: the
  `start_training_clock()` call, the `try … except TrainingTimeUp` around the training loop,
  and the final `evaluate_bpb` + `val_bpb:` print. They make every run hard-bounded to the
  per-experiment budget and guarantee a graceful final eval — removing them just makes your
  own run overrun or crash without a score.

---

**To run:** the orchestrator follows [`ENGINE.md`](ENGINE.md) — bring up the box with
`vast.py start`, then each round edit every slot's `train.py` and run them all in parallel with
`python vast.py round` (the orchestrator runs the experiments itself), alongside one
research-scout subagent, and loop. The whole control plane is `python vast.py --help`.
