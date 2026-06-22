# autoresearch — parallel LLM-training research on a Vast GPU box

This is an autonomous research org whose **one job** is to find the `train.py` that
reaches the **lowest held-out `val_bpb`** within a fixed per-run training budget. The
research loop and the agents run **locally on this laptop**; the GPU work (`train.py`,
the fixed training budget) runs on **one rented Vast.ai box per session** with several
GPUs. A pool of subagents runs experiments **in parallel** — one per GPU — under a
**tightly steered round loop** so their work never overlaps and always compounds.

The control plane is one file: **`vast.py`** (run `python vast.py --help`). It
rents/destroys the box, prepares it, benchmarks the GPUs, runs experiments on a chosen
GPU slot (with a hard time cap + ghost-run reaping), and reaps stray runs. You (the
human) iterate on **this file** (`program.md`) and on `train.py`. `prepare.py` is fixed.

**The metric.** `val_bpb` = bits-per-byte on a held-out shard. **Lower is better**,
vocab-size-independent so architecture changes compare fairly. Every run also prints
`peak_vram_mb` (watch it — see VRAM below).

**What's editable.** `train.py` — everything is fair game: architecture, optimizer,
hyperparameters, training loop, batch size, model size. **Not** editable: `prepare.py`
(fixed regime, data, tokenizer, time budget, seq len) and the `evaluate_bpb` metric.

---

## How the round loop works (the core idea — read this carefully)

**You are the orchestrator. You never run experiments yourself.** You dispatch them to
subagents and keep the global picture. The loop is **short, tightly steered, and
non-overlapping**:

> You choose, per round, whether every subagent runs **1 or 2 experiments** — the SAME
> count for all slots that round — then each subagent **runs exactly that many, STOPS,
> and returns** its findings. You dispatch all slots **in the background**, then use the
> window while they run to (a) analyze the previous round and (b) **research improvements
> with the `research` skill** (papers/web). When they return, you fold in results +
> research, **stack every confirmed win into the champion**, and hand each new subagent a
> **specific, disjoint direction** for its next experiments. Then repeat — forever, with
> the champion strictly improving over time.

**Choosing 1 vs 2 experiments this round (your call, uniform across slots):**
- **1 each** when you want **tight steering** — early on while you're still calibrating
  which axes pay off, when ideas are uncertain/risky, or when little time remains before
  the deadline (so a half-finished round doesn't strand work). You re-plan twice as often.
- **2 each** when an axis is **promising and worth depth** — let a slot try an idea and a
  natural follow-up in one dispatch, so you spend more of your window on research and less
  on re-dispatching. Default to 2 once the search is flowing.

Each round, in order:

1. **Pick the champion** = lowest `val_bpb` so far (its `train.py` lives on the champion
   branch `autoresearch/<tag>`). Every subagent starts its experiments from the champion.

2. **Assign each slot a DISJOINT axis — non-overlap is a HARD RULE.** Two slots must
   never be able to try the same change. Vague "themes" overlap, so DON'T partition by
   theme. Partition by **which component of `train.py` the change touches**. Each idea
   belongs to exactly one axis, so distinct axes cannot collide:

   - **Axis A — Optimizer & LR schedule.** Muon/AdamW hyperparams (lr, betas, eps, weight
     decay, momentum), warmup/decay shape, LR-vs-width scaling, grad clipping.
   - **Axis B — Attention.** `n_head`, `n_kv_head` (GQA), `window_pattern`
     (local/global mix), rotary base/θ, attention scale / logit softcap, QK-norm.
   - **Axis C — MLP, activation & normalization.** MLP ratio / hidden dim, activation
     function, norm type & placement (pre/post, RMS), residual scaling.
   - **Axis D — Capacity & token budget.** `DEPTH`/`n_layer`, `n_embd`,
     `DEVICE_BATCH_SIZE` / `TOTAL_BATCH_SIZE` / grad-accum, precision/dtype, packing.
     (This axis OWNS all batch/size changes — and it's the VRAM-sensitive one.)
   - **Axis E — Embeddings, init & regularization.** Value embeddings, weight tying,
     embedding/init scale, logit/z-loss, dropout, init schemes.

   **Pre-assign centrally, BEFORE spawning** (this is what makes non-overlap airtight —
   agents never pick from a shared menu at the same instant and collide). For each slot
   write down: its single axis, the **concrete idea(s) it owns this round** (one if it's a
   1-experiment round, two if a 2-experiment round), and an explicit **OFF-LIMITS list** =
   every other slot's axis + ideas. With N slots, give the first N distinct axes;
   **rotate axes across rounds** so each slot sees variety. The off-limits list reduces
   collisions but doesn't *guarantee* them — subagents sometimes reach for the "obvious"
   first idea regardless — so for a collision-prone idea (or any slot that strayed last
   round), **don't name the idea, specify the exact change** (the code/diff) so there's
   nothing to substitute. Compliance is verified on return (step 5).

3. **Keep a "Tried" registry in `findings.md` so nothing repeats across rounds.** Every
   concrete idea ever attempted (+ its result) goes on the list. Before assigning a round,
   read it and never re-assign anything already there.

4. **Dispatch in the background, then analyze + research while they run.** Spawn the slot
   subagents (one per slot) as **background** agents in a single batch so you aren't
   blocked. Then spend the ~10-minute window: digest the last round's numbers, and run the
   **`research` skill** on the open question (e.g. "efficient small-LLM pretraining tricks:
   optimizer, attention, normalization, init"). Fold what you learn into the next round's
   per-slot directions.

5. **When they return, VERIFY, then STACK the genuine wins.** Read their summaries, but
   reconcile against ground truth before trusting anything — then update `findings.md`
   yourself. The champion grows by accumulating every *real* improvement, not by replacing
   one good idea with another:
   - **The ledger is ground truth; verify compliance.** `results.tsv` (and each slot's git
     commits/diff) is authoritative — a subagent's self-report can be wrong or incomplete: a
     slot may have run more than {k} experiments, strayed outside its axis, or — worst —
     duplicated another slot's idea (this happens even with an explicit off-limits list).
     Reconcile every reported result against `results.tsv` and the slot's commits. If a slot
     exceeded {k}, went off-axis, or overlapped another slot, **treat the ledger as truth,
     discard the violating/duplicate runs**, and tighten that slot next round — hand it a
     **fully-specified change (exact code/diff), not a menu**, so there's nothing left to
     substitute.
   - **You are the sole writer of `findings.md`.** Update it every round, right here, from
     the reconciled results (Champion + Tried + Dead ends). Subagents never touch it, so it
     stays clean and race-free.
   - **Confirm before counting it.** A "win" is only genuine if it survives a re-run on the
     same slot (gains in this regime can be noise). Re-run a promising delta once; if it
     holds, it's real.
   - **Stack across axes.** Because the axes are disjoint, wins from different axes usually
     compose (an optimizer win + an attention win + an init win). Fold each confirmed win
     into the champion `train.py`, then **re-run the stacked champion once** to verify the
     combination still improves — occasionally two changes interact and cancel; if so, keep
     the better single one and note the conflict in `findings.md`.
   - **Then search on top of the new champion.** Every subsequent round starts from the
     stacked champion and looks for the *next* improvement, so `val_bpb` keeps dropping over
     the whole session instead of plateauing. Design the next round (research-informed,
     disjoint axes) from this champion.

6. **Reap between rounds.** Run `python vast.py reap` to kill any stray run and confirm the
   GPUs are idle before dispatching the next round (no ghosts, no leftover VRAM).

7. **Loop forever** (see **NEVER STOP**).

### The prompt to give each slot subagent

> You are a research subagent on **slot {i} (GPU {i})**. Work ONLY in
> `worktrees/slot{i}` on branch `autoresearch/<tag>-slot{i}`. Your axis this round is
> **{axis}** and the ONLY ideas you may try are: **{owned_ideas}** (exactly {k}). These are
> **OFF-LIMITS** (other slots own them — never try them): **{off_limits}**. Already tried
> (do not repeat): **{tried_digest}**. The current champion is `val_bpb={champion}` — its
> `train.py` is on branch `autoresearch/<tag>`; start from it
> (`git -C worktrees/slot{i} checkout autoresearch/<tag> -- train.py`).
>
> Run **exactly {k} experiment(s), then STOP and return** to the orchestrator (it will
> give you your next direction — do NOT exceed {k}). Before each, run
> `python3 vast.py status`; if under ~9 minutes remain to the deadline, stop early and
> return what you have. For each of the {k}:
> 1. Edit `worktrees/slot{i}/train.py` with one of YOUR {k} assigned idea(s) (axis {axis}
>    only). Change only what that idea needs — keep everything else at the champion's
>    values so the delta is attributable.
> 2. `git -C worktrees/slot{i} add -A && git -C worktrees/slot{i} commit -m "<idea>"`.
> 3. Run it: `python vast.py exp --slot {i} --train worktrees/slot{i}/train.py`. This
>    syncs your train.py, runs on GPU {i} (CPU-pinned), and **hard-caps the run** so it
>    can't overrun the budget or hang. Read the printed `val_bpb` / `peak_vram`.
>    - Empty / `CRASH` / `TIMEOUT` / `OOM` = it failed. If `OOM`, your idea used too much
>      VRAM — log it `crash`, `git reset --hard HEAD~1`, and DON'T retry it bigger. (If you
>      must keep the idea, the only fix is a smaller `DEVICE_BATCH_SIZE` — but that's Axis D,
>      not yours, so just report it and move on.)
> 4. Log it to the ledger: `python vast.py log <commit> <val_bpb> <mem_gb>
>    <keep|discard|crash> slot{i} "<description>"` (atomic — safe under concurrency). This
>    `results.tsv` ledger is the ground-truth record and the ONLY shared file you write.
> 5. If `val_bpb` improved, keep the commit. If equal/worse/failed,
>    `git -C worktrees/slot{i} reset --hard HEAD~1`.
>
> Do **NOT** edit `findings.md` — the orchestrator is its sole writer (it curates it from
> your returned summary; this prevents concurrent-write clobbering). Stay on your
> slot/worktree/branch only — never touch another slot's GPU, worktree, or branch, and never
> run more than {k} experiments. When done, **return a structured summary**: for each
> experiment — the exact one-line diff you made (so the orchestrator can verify it stayed in
> axis {axis}), its commit hash, `val_bpb` / `peak_vram` (or the CRASH/OOM/TIMEOUT reason),
> and whether to keep it or put it on the Tried list.

This keeps slots **non-interfering** (distinct GPU + CPU cores + worktree + branch +
remote dir) yet **interconnected** (the shared `results.tsv` ledger they write + the
orchestrator-curated `findings.md` digest they read + one champion they all build from).

---

## Session setup

When the human says "let's kick off a new session", do this once, in order:

1. **Ask how long to run** (default 3h). This becomes the box's hard auto-destroy deadline.
2. **Agree on a run tag** from today's date (e.g. `mar5`); `git checkout -b autoresearch/<tag>`
   from master (the champion branch). It must not already exist.
3. **Read the in-scope files**: `README.md`, `prepare.py` (fixed; note `TIME_BUDGET`), and
   `train.py` (the file you edit — skim the model, optimizer, and training loop).
4. **Rent the box** (refuses anything over the per-GPU cap; prints the rate):
   `python vast.py up --gpus 4 --hours <H> --max-price 0.60`. Fall back to `--gpus 2` if no
   4-GPU offer qualifies.
5. **Start the watchdog in the background immediately**: `python vast.py watchdog` (so the
   box can never outlive its deadline).
6. **Prepare the box**: `python vast.py setup` (uploads the repo, uses the template's
   preinstalled torch + light deps — no torch download — preps data, detects GPU/CPU).
7. **Bench for objectivity** (once): `python vast.py bench`; read `bench.json` (GPU spread
   small ⇒ comparable; high concurrency tax ⇒ use fewer slots).
8. **Read the slot count**: `python vast.py status` prints `num_gpus` = max parallel slots
   (call it `N`).
9. **One worktree + branch per slot**:
   `for i in 0..N-1: git worktree add worktrees/slot$i -b autoresearch/<tag>-slot$i`.
10. **Initialize shared state** (both untracked/gitignored): `results.tsv` (auto-created by
    `vast.py log`; the append-only ledger the subagents write) and `findings.md`
    (**orchestrator-owned**; seed sections **Champion**, **Tried**, **Dead ends** with
    "baseline pending").
11. **Establish the baseline AND confirm it fits VRAM.** Run the unmodified `train.py` on
    slot 0: `python vast.py exp --slot 0 --train train.py`.
    - If it returns a `val_bpb` and `peak_vram` is comfortably under the GPU's memory
      (e.g. < ~22 GB on a 24 GB 4090) → log it `keep`, copy `train.py` onto the champion
      branch, record baseline `val_bpb` in `findings.md`, and begin the loop.
    - If it **OOMs** → the baseline is too big for this GPU. Lower `DEVICE_BATCH_SIZE` (then
      `DEPTH`/`n_embd` if needed) until it fits with headroom, re-run, and make THAT the
      champion baseline. Do this **before** any round so no slot inherits an OOM.
12. **(Optional) Offer the dashboard**: `python vast.py dashboard` — a localhost page (val_bpb
    chart, leaderboard, recent runs, box cost/deadline) that auto-refreshes every 5s.

---

## The experiment (what runs on the GPU)

Each experiment is `train.py` for the **fixed training budget** (`TIME_BUDGET` in
`prepare.py`, wall clock, excluding startup/compile). `vast.py exp` syncs the file, pins
the GPU + CPU cores, runs it, and parses the result.

**Each run is bounded by the budget — without a brutal external kill.** `train.py`
self-stops at `TIME_BUDGET` (it checks elapsed training time and breaks), so a healthy run
ends on its own at the set budget plus a little compile/eval overhead. `vast.py exp` does
**not** force-kill the run. It only reaps any *ghost* still on that slot **before**
launching (clearing leftovers from a died subagent) so a new run never collides with a
stale one. If you ever see a genuinely stuck run, clear it with `python vast.py reap`.

`train.py` prints a summary; the key line is `val_bpb:`. `exp` extracts it and prints
`RESULT_JSON:{...}` (or a `CRASH`/`TIMEOUT`/`OOM` reason).

---

## VRAM & OOM discipline (no OOMs)

- **Keep peak VRAM well under the GPU's limit** (on a 24 GB 4090, stay under ~22 GB). The
  baseline is sized to fit; only **Axis D** changes batch/model size.
- An **OOM is a failure**: log it `crash`, `git reset --hard`, and don't retry the same
  idea larger. Multi-pass or wider ideas that need more memory must be paired with a
  smaller `DEVICE_BATCH_SIZE` — which is **Axis D's** job, so leave it to that slot.
- If you ever suspect a stuck/ghost run is holding GPU memory, `python vast.py reap` clears
  stray runs and shows what's still on each GPU. Reap between rounds as a habit.

---

## Logging results

Log every experiment via `python vast.py log` (locks the file; concurrency-safe). 6 cols:

```
commit   val_bpb   memory_gb   status   branch   description
```

git commit (7 chars; `0000000` on fail) · `val_bpb` (`0.000000` on fail) · peak GB `.1f` ·
`keep`/`discard`/`crash` · slot (e.g. `slot2`) · short description. Do **not** `git add`
`results.tsv` or `findings.md` — they stay untracked.

---

## Objectivity

The fixed budget makes runs comparable — same wall-clock GPU time ⇒ a lower `val_bpb` is a
genuinely better config for this platform. `bench` confirms the GPUs are equal (spread)
and measures the concurrency tax; `vast.py` pins disjoint CPU cores per slot. When a
`val_bpb` delta is small, **re-run on the same slot before crowning a champion** — gains
can be noisy.

---

## Sharing findings

`findings.md` is the swarm's shared brain, and **the orchestrator is its sole writer.**
Subagents return summaries and log raw runs to the `results.tsv` ledger; only you edit
`findings.md`, so it never gets clobbered by concurrent appends and stays a curated record
rather than a pile of overlapping notes. **Update it every round, immediately after you
evaluate the handed-back results** (reconciled against `results.tsv`). Keep three sections
current: **Champion** (val_bpb + the one-line change that achieved it), **Tried** (every
idea + result, so nothing repeats), and **Dead ends** (consistently-failing ideas). Give
subagents the latest digest each round.

---

## Teardown

- The watchdog auto-destroys the box at the deadline no matter what.
- End early / when the human stops: `python vast.py down`, then `python vast.py ps` to
  confirm nothing is billing. `extend --hours <H>` pushes the deadline. `ps`/`nuke` catch
  and kill orphans. `reap` clears stray runs without destroying the box.

---

## NEVER STOP

Once the loop has begun, do NOT pause to ask whether to continue. The human may be asleep
and expects research **until the deadline, an interrupt, or teardown**. When a round
finishes, immediately design and dispatch the next — keep all GPUs busy every minute. If
ideas run dry, think harder: re-read `train.py`, mine the literature with the `research`
skill, combine near-misses across axes, try more radical changes, rotate axes. The loop
runs until stopped, period.
