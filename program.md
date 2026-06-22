# autoresearch — parallel research on a Vast GPU box

This is an experiment to have the LLM do its own research. The research loop and the
agents run **locally on this laptop**; the GPU work (`train.py`, the fixed 5-minute
budget) runs on **one rented Vast.ai box per session** with several GPUs. A pool of
subagents runs experiments **in parallel** — one per GPU — sharing findings so they
compound instead of colliding.

The whole control plane is one file: **`vast.py`** (run `python vast.py --help`). It
rents/destroys the box, prepares it, benchmarks the GPUs, and runs experiments on a
chosen GPU slot. You (the human) iterate on **this file** (`program.md`) and on
`train.py`. You never touch `prepare.py`.

---

## Session setup

When the human says "let's kick off a new session", do this once, in order:

1. **Ask how long to run.** Ask the human, approximately how many hours this session
   should run (default 3). This becomes the box's hard auto-destroy deadline — the box
   can never outlive it.

2. **Agree on a run tag.** Propose one from today's date (e.g. `mar5`). The branch
   `autoresearch/<tag>` must not exist yet (fresh run). Create it from master:
   `git checkout -b autoresearch/<tag>`. This is the **champion** branch (the current
   best `train.py`).

3. **Read the in-scope files** for full context: `README.md`, `prepare.py` (fixed
   constants, data, tokenizer, dataloader, the `evaluate_bpb` metric — read-only) and
   `train.py` (the model/optimizer/loop you will edit).

4. **Rent the box** (cheapest qualifying multi-GPU box; refuses anything over the
   per-GPU price cap and prints the bill rate):
   ```
   python vast.py up --gpus 4 --hours <H> --max-price 0.60
   ```
   If no 4-GPU offer qualifies, fall back to `--gpus 2`. Tell the human the rate.

5. **Start the watchdog in the background immediately** so the box can never be left
   billing past the deadline:
   ```
   python vast.py watchdog        # run in background
   ```

6. **Prepare the box** (uploads the repo, uses the template's **preinstalled** torch and
   installs only the light deps — no multi-GB torch download — then downloads data +
   trains the tokenizer and detects GPU/CPU topology). One-time:
   ```
   python vast.py setup
   ```

7. **Benchmark for objectivity** (REQUIRED once per session). This confirms the GPUs
   are interchangeable and measures any throughput tax from running all of them at once:
   ```
   python vast.py bench
   ```
   Read `bench.json`. If "GPU spread" is small (≤5%) the GPUs are equivalent → results
   are comparable across slots. If the "concurrency tax" is high (>15%), prefer fewer
   parallel slots so comparisons stay clean (see **Objectivity** below).

8. **Read the slot count.** `python vast.py status` prints `num_gpus` = the number of
   GPUs = the max number of parallel slots. Call it `N`.

9. **Create one worktree + branch per slot** so the slots never collide in git:
   ```
   for i in 0..N-1:  git worktree add worktrees/slot$i -b autoresearch/<tag>-slot$i
   ```

10. **Initialize shared state** (both untracked, gitignored):
    - `results.tsv` — the ledger (created automatically by `vast.py log`).
    - `findings.md` — the shared knowledge log: what's been tried, what worked, what
      failed, and the current champion. Seed it with "baseline pending".

11. **Establish the baseline.** Run the unmodified `train.py` once on slot 0:
    ```
    python vast.py exp --slot 0 --train train.py
    ```
    Log it as the baseline (status `keep`), copy this `train.py` into the champion
    branch, and record the baseline `val_bpb` in `findings.md`. Now begin the loop.

12. **(Optional) Offer the live dashboard.** Tell the human they can watch progress in a
    browser: `python vast.py dashboard` (opens a localhost page that auto-refreshes
    every 5s — val_bpb chart, leaderboard, recent runs, and the box's cost/deadline).
    It reads the local `results.tsv` + `.vast_state.json`, so it updates as slots log.

---

## How parallelism works (orchestrator + slot subagents)

**You are the orchestrator.** You do not run experiments yourself; you dispatch them
to subagents and keep the global picture. Each round:

1. **Pick the champion** = lowest `val_bpb` so far (its `train.py` lives on the
   champion branch `autoresearch/<tag>`).

2. **Assign each free slot a distinct focus** so subagents in the same round never try
   the same thing. Rotate focuses across rounds. Example split for 4 slots:
   - slot0 → optimizer (LR schedule, Muon/AdamW params, betas, weight decay)
   - slot1 → architecture (depth/width, heads, MLP ratio, norm placement, activation)
   - slot2 → data/throughput (batch size, sequence packing, grad-accum, dtype)
   - slot3 → regularization & init (dropout, init scale, embedding tricks, value embeds)

3. **Spawn the subagents in a single message** (so they run concurrently — one per
   slot). Give each subagent the prompt below, filled in with its slot index, its
   focus, the current champion `val_bpb`, and the latest `findings.md` digest.

4. **When they return**, read their summaries, update `findings.md`, and **cross-
   pollinate**: take the round's best result; if it beats the champion, copy that
   slot's `train.py` onto the champion branch and announce the new champion in
   `findings.md`. Then start the next round from the new champion.

5. **Loop forever** (see **NEVER STOP**).

### The prompt to give each slot subagent

> You are a research subagent on **slot {i} (GPU {i})**. Work ONLY in
> `worktrees/slot{i}` on branch `autoresearch/<tag>-slot{i}`. Your focus this round
> is **{focus}**. The current champion is `val_bpb={champion}` — its `train.py` is on
> branch `autoresearch/<tag>`; start from it (`git -C worktrees/slot{i} checkout
> autoresearch/<tag> -- train.py`). Read `findings.md` first so you don't repeat what's
> already been tried.
>
> Run a short inner loop of **3–6 experiments**:
> 1. Edit `worktrees/slot{i}/train.py` with one concrete idea in your focus area.
> 2. `git -C worktrees/slot{i} add -A && git -C worktrees/slot{i} commit -m "<idea>"`.
> 3. Run it on your slot (this syncs your train.py to the box and runs on GPU {i},
>    CPU-pinned): `python vast.py exp --slot {i} --train worktrees/slot{i}/train.py`.
> 4. Read the printed `val_bpb` / `peak_vram`. Empty/CRASH = it failed.
> 5. Log it: `python vast.py log <commit> <val_bpb> <mem_gb> <keep|discard|crash>
>    slot{i} "<description>"` (atomic — safe under concurrency).
> 6. If `val_bpb` improved, keep the commit (advance your branch). If equal/worse,
>    `git -C worktrees/slot{i} reset --hard HEAD~1`.
> 7. Append a one-line note to `findings.md` (what you tried + the result).
>
> Stay on your slot and your worktree only — never touch another slot's GPU, worktree,
> or branch. When done, return a short summary: your best `val_bpb`, the winning diff
> idea, and which ideas flopped (so others don't repeat them).

This keeps slots **non-interfering** (distinct GPU + CPU cores + worktree + branch +
remote dir) yet **interconnected** (shared `findings.md`, shared ledger, a single
champion they all build from).

---

## The experiment (what actually runs on the GPU)

Each experiment is `train.py` running for the **fixed 5-minute training budget** (wall
clock, excluding startup/compile). `vast.py exp` handles syncing the file, pinning the
GPU + CPU cores, running it, and parsing the result. The metric is **`val_bpb`**
(validation bits per byte) — **lower is better**, vocab-size-independent so
architectural changes compare fairly.

**What you CAN do:** edit `train.py` only — architecture, optimizer, hyperparameters,
training loop, batch size, model size. All fair game.

**What you CANNOT do:**
- Modify `prepare.py` (read-only: fixed eval, data, tokenizer, time budget, seq len).
- Add dependencies beyond `pyproject.toml`.
- Modify the evaluation harness (`evaluate_bpb` is ground truth).

**VRAM** is a soft constraint — modest increases are fine for real `val_bpb` gains,
but don't blow up. On a 4090 (24 GB) keep peak well under that.

**Simplicity criterion:** all else equal, simpler is better. A tiny gain that adds
ugly complexity isn't worth it; a simplification that holds or improves `val_bpb` is a
great outcome.

`train.py` prints a summary; the key line is `val_bpb:`. `vast.py exp` extracts it for
you and prints `RESULT_JSON:{...}`.

---

## Logging results

Log every experiment to `results.tsv` via `python vast.py log` (it locks the file, so
concurrent slots never corrupt it). Tab-separated, 6 columns:

```
commit   val_bpb   memory_gb   status   branch   description
```

1. git commit (short, 7 chars) — `0000000` for a crash
2. `val_bpb` (e.g. `1.234567`) — `0.000000` for a crash
3. peak memory GB, `.1f` (peak_vram_mb / 1024) — `0.0` for a crash
4. status: `keep`, `discard`, or `crash`
5. branch/slot (e.g. `slot2`)
6. short description (commas are fine; it's one quoted arg)

Do **not** `git add` `results.tsv` or `findings.md` — they stay untracked.

---

## Objectivity (this is the point of the bench)

The 5-minute budget is what makes runs comparable — every experiment gets the same
wall-clock GPU time, so a better `val_bpb` means a genuinely better config for this
platform. Two things protect that under parallelism, both handled for you:

- **Equal GPUs.** All slots are the same GPU model on one box; `bench` confirms their
  solo throughput matches (the "GPU spread"). If the spread is large, treat cross-slot
  comparisons cautiously and re-bench / re-rent.
- **No hidden contention.** The dataloader tokenizes on the CPU, so concurrent runs
  could steal each other's throughput. `vast.py` pins each slot to disjoint CPU cores
  (`taskset`) to prevent this, and `bench` measures the residual "concurrency tax".
  - tax ≤5%: full parallel is objective.
  - 5–15%: fine, but only compare experiments that ran at the **same** parallelism.
  - >15%: drop to fewer parallel slots for clean comparisons.

When in doubt about whether a small `val_bpb` delta is real, re-run it on the same slot
under the same conditions before declaring a winner.

---

## Sharing findings

`findings.md` is the swarm's shared brain. Keep it current:
- A running list of ideas tried (slot, idea, result, keep/discard).
- The current champion `val_bpb` and the one-line diff that achieved it.
- "Dead ends" — things that consistently failed, so no slot re-tries them.

Before each round, give subagents the latest digest. After each round, fold their
summaries in and update the champion. This is how parallel work compounds instead of
duplicating.

---

## Teardown

- The watchdog auto-destroys the box at the deadline no matter what.
- To stop early or when the human ends the session: `python vast.py down` (destroys the
  box and clears state). Then `python vast.py ps` to confirm nothing is left billing.
- `python vast.py extend --hours <H>` pushes the deadline out if the human wants longer.
- If anything looks orphaned: `python vast.py ps` lists ALL instances; `nuke` kills all.

---

## NEVER STOP

Once the loop has begun (after setup), do NOT pause to ask the human whether to
continue. The human may be asleep and expects you to keep researching **until the
deadline, the human interrupts, or the box is destroyed**. When a round finishes, start
the next one. If ideas run dry, think harder: re-read `train.py` and `prepare.py` for
new angles, read papers referenced in the code, combine previous near-misses, try more
radical architectural changes, re-assign slot focuses. Keep the GPUs busy. The loop
runs until stopped, period.
