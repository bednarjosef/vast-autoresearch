<!-- ============================================================================
     ENGINE.md — the fixed autoresearch engine. DO NOT EDIT.

     This file is the GENERAL machinery: how the orchestrator and subagents run a
     session, regardless of WHAT is being researched. The "what / how to optimize"
     lives in program.md (the mission), which IS meant to be edited. Agents and the
     human edit program.md and train.py — never this file.
     ============================================================================ -->

# The autoresearch engine (fixed — do not edit)

`program.md` says **what** to optimize and the axes to explore. **This file says how the
machine runs.** Read `program.md` first (the mission + config), then run the loop below.

The control plane is one file: **`vast.py`** (`python vast.py --help`). It rents/destroys
the box, prepares it, benchmarks the GPUs, and runs one experiment on a chosen GPU slot.

---

## Roles — and the two rules that are non-negotiable

**You (the main agent) are the ORCHESTRATOR.** You plan, dispatch, analyze, and curate.
Experiments run on the GPUs via **subagents** you spawn — one per GPU slot.

1. **YOU MUST RUN N SUBAGENTS — NOT N EXPERIMENTS YOURSELF — ALL IN THE FOREGROUND.** Launch
   experiments by **spawning subagents with the Agent tool, one per GPU slot, all in a single
   message** so they run **concurrently in the foreground**. **Wait for the whole batch to
   return — never run the subagents in the background, and never poll for them.** A single
   message with N Agent calls blocks exactly once, on the whole batch; that is correct. (A
   background-and-poll pattern is what makes orchestration get stuck.) **Never call
   `python vast.py exp` yourself** — if you're about to run an experiment directly, STOP and
   spawn a subagent. The entire point is N GPUs in parallel; serial experiments in the main
   thread collapse the design. With a 4-GPU box, every round spawns 4 subagents.

2. **EXPERIMENTS ARE SYNCHRONOUS — NEVER POLL OR BUILD A WAIT-LOOP.** `vast.py exp` blocks
   until the run finishes (the full training budget + compile/eval) and then **prints the
   result** (`val_bpb` / `RESULT_JSON`, or a `CRASH`/`OOM` reason). A subagent simply calls
   it in the **foreground** and lets it return. **Do NOT** background it, redirect it to a
   log and poll the log, set up a "wait loop", or repeatedly read files waiting for it to
   finish. One foreground `exp` call returns one finished experiment with its result in the
   output. (Polling here is the #1 way subagents get stuck — don't.)

3. **DURING RESEARCH, ONLY `train.py` MAY CHANGE — NO CHEATING.** Once the loop is running,
   neither you nor any subagent may edit `prepare.py`, `evaluate_bpb`, the `forward`→logits
   contract, or `ENGINE.md`. Every gain must come from `train.py` alone, so the score can't be
   gamed. This is enforced structurally: `vast.py exp` uploads **only** `train.py` to the box,
   so the box's `prepare.py` and metric — frozen when you ran `start`/`setup` — score every run
   no matter what a subagent edits locally. (The human or their agent MAY edit `prepare.py` to
   set the regime **before** launching research; during the loop it stays frozen.)

---

## Session bring-up

Do this once, when the human starts a session:

1. **Pick a per-experiment budget and session length** (from `program.md`'s config, or ask).
2. **Bring up the box in ONE command** — rent → watchdog → setup → bench:
   ```
   python vast.py start --gpus 4 --hours <H> --minutes <M> --max-price 0.60
   ```
   `start` rents the cheapest qualifying box, launches the deadline **watchdog in the
   background** (the box can never outlive `--hours`), prepares it (template torch + light
   deps — no torch download — plus data/tokenizer), and runs the **~1-min bench** (which
   also warms each slot's compile cache so first experiments start fast). If no 4-GPU offer
   qualifies, retry with `--gpus 2`.
3. **Read the slot count**: `python vast.py status` prints `num_gpus` = the number of GPUs
   = max parallel slots (call it `N`), plus the per-experiment budget and cost/deadline.
4. **One worktree + branch per slot** (so slots never collide in git):
   `for i in 0..N-1: git worktree add worktrees/slot$i -b autoresearch/<tag>-slot$i`
   (`<tag>` from today's date; create the champion branch `autoresearch/<tag>` from master
   first).
5. **Init shared state** (both untracked/gitignored): `results.tsv` (auto-created by
   `vast.py log`; the append-only ledger subagents write) and `findings.md`
   (**orchestrator-owned**; seed sections **Champion**, **Tried**, **Dead ends**).
6. **Establish the baseline + confirm it fits VRAM.** Run the unmodified `train.py` on slot
   0 ONCE — do this yourself, just to seed the champion (this is the only `exp` you run; all
   research experiments go through subagents):
   `python vast.py exp --slot 0 --train train.py`.
   - Returns a `val_bpb` and `peak_vram` comfortably under the GPU (e.g. < ~22 GB on a 24 GB
     4090) → log `keep`, copy `train.py` onto the champion branch, record it in
     `findings.md`, begin the loop.
   - **OOM** → baseline too big. Lower `DEVICE_BATCH_SIZE` (then `DEPTH`/`n_embd`) until it
     fits with headroom; make THAT the champion baseline before any round.
7. **(Optional) Offer the dashboard**: `python vast.py dashboard` (localhost, auto-refresh).

---

## The round loop

**You choose, per round, whether every subagent runs 1 or 2 experiments** — the SAME count
for all slots that round. Each subagent runs exactly that many, **then STOPS and returns**.
You spawn all slots in **one foreground, concurrent batch** and wait for it to return (add a
"research scout" subagent to the same batch to mine the literature with the **`research`
skill** while the GPUs are busy — see step 4). When the batch returns you **compound** the
confirmed wins into the champion and dispatch the next round. Repeat — forever, the champion
strictly improving by building UPON itself.

- **1 each** for tight steering (early calibration, risky ideas, or little time left).
- **2 each** to go deeper on a promising axis (idea + a natural follow-up in one dispatch).

> **USE THE `research` SKILL — every round, not from memory.** Before assigning directions,
> mine the literature with the **`research` skill** (scholarly + web): find genuinely **novel
> ideas**, the **current SOTA** for whatever you're optimizing, and evidence on whether an idea
> is known to work or to fail (so you don't burn a GPU rediscovering it). Ground each slot's
> direction in what you find. The cheapest way to do this without idling is to spawn a
> **research-scout subagent in the same foreground batch** as the experiments (step 4). Lean on
> the skill hard — it is your primary source of ideas, not your prior knowledge.

Each round, in order:

1. **Pick the champion** = lowest `val_bpb` so far (its `train.py` is on `autoresearch/<tag>`).
   Every subagent starts from it.

2. **Assign each slot a DISJOINT axis — non-overlap is a HARD RULE.** Use the axes defined
   in `program.md`. Each idea belongs to exactly one axis, so distinct axes cannot collide.
   **Pre-assign centrally, BEFORE spawning**: for each slot write down its single axis, the
   concrete idea(s) it owns this round (1 or 2, matching the count), and an explicit
   **OFF-LIMITS list** = every other slot's axis + ideas. Give the first N distinct axes;
   **rotate axes across rounds**. The off-limits list reduces collisions but doesn't
   *guarantee* them — for a collision-prone idea, or any slot that strayed last round,
   **don't name the idea, specify the exact code/diff** so there's nothing to substitute.

   **BREADTH, NOT TUNNEL VISION — the rule most often broken.** A round of N slots covers N
   *different* axes. **Never put two slots on the same idea or close variants** — e.g. "tune
   flooding `b`", "per-sequence flooding", and "find the perfect flooding value" are ONE idea
   family, so that's at most ONE slot, never four. And **never spend consecutive rounds
   circling one discovery.** The moment an idea is confirmed and banked into the champion
   (step 5) it is **DONE**: it moves to the **Banked** list and is OFF the menu — stop
   assigning slots to perfect it. Chasing the "ideal" value of an already-good hyperparameter
   (3 vs 3.25 vs 3.5) is diminishing-returns busywork: do it **once**, as a single sweep on
   **one** slot, bank the best value, and spend every other slot on **axes you have NOT
   explored yet**. Your job is to keep finding NOVEL wins on top of the champion — not to
   polish one. Each round, glance at the coverage in `findings.md` and prioritize the
   **least-explored** axes; if you catch yourself assigning >1 slot to the current hot idea,
   stop and re-diversify.

3. **Keep a "Tried" registry AND a "Banked" list in `findings.md`.** Tried = every idea
   attempted (so nothing repeats); Banked = wins already folded into the champion (so they're
   never re-explored). Read both before assigning; **never re-assign anything on either list**,
   and never assign a close variant of a Banked idea.

4. **Spawn N subagents in ONE message — FOREGROUND, concurrent — and wait for the batch.**
   Use the Agent tool: one subagent per slot, the prompt below filled in, all in a single
   message so they run in parallel. They run in the **foreground**; the call returns when all
   slots have finished. **Never background the subagents or poll for them.** To mine the
   literature *while the GPUs are busy* without backgrounding anything, add **one extra
   subagent to the same batch** — a "research scout" that runs the **`research` skill** (or
   web search) on the open question and returns fresh ideas alongside the experiment results.
   (Alternatively, do the research yourself between rounds — just never via a background/poll
   loop.) Fold what the scout finds into the next round's directions.

5. **When they return, VERIFY against ground truth, then COMPOUND the genuine wins.** The
   champion grows by **accumulating** confirmed improvements — you build UPON it, you don't
   restart from scratch each round.
   - **The ledger is ground truth.** `results.tsv` + each slot's git commits/diffs are
     authoritative; a subagent's self-report can be wrong. Reconcile every reported result.
     If a slot ran more than its count, went off-axis, or duplicated another slot, **trust
     the ledger, discard the violating/duplicate runs**, and next round hand that slot a
     **fully-specified diff, not a menu**.
   - **You are the sole writer of `findings.md`.** Update it every round from the reconciled
     results (Champion + Tried + **Banked** + Dead ends). Subagents never touch it.
   - **Confirm before counting a win.** Gains can be noise — re-run a promising delta once;
     if it holds, it's real.
   - **Bank a win, then LEAVE IT ALONE.** The moment a win is folded into the champion, add it
     to the **Banked** list and stop assigning slots to it. At most ONE follow-up may fine-tune
     its hyperparameter — a single sweep on one slot — after which the value is frozen.
     Re-spending slots to "perfect" a banked idea (sweeping its value, per-token/per-seq
     variants, etc.) is the #1 way the loop stalls. Keep what worked; go find the NEXT new win.
   - **Compound: find A, find B, then TEST A+B and adopt the best as the new base.** When two
     (or more) confirmed wins land — especially on different axes, which usually compose —
     build the combination on top of the champion and **run it** (`exp` on the merged
     `train.py`). If the combo beats every part, *it* becomes the new champion and the next
     round builds on the combo. If the combo is worse than its best component (the changes
     interact/cancel), keep the better single one and record the conflict in `findings.md`.
     Keep folding every confirmed win in, so the champion is always "everything that's worked
     so far," and `val_bpb` keeps dropping all session.
   - **Build UPON, don't restart.** Every subagent starts from the current champion (which
     already contains all stacked wins), so each experiment is "champion + one new idea" by
     construction. Only occasionally test an idea against the bare baseline — e.g. to isolate
     whether a stacked change is now *blocking* a bigger gain, or to ablate an interaction.

6. **Reap between rounds**: `python vast.py reap` kills any stray run and confirms the GPUs
   are idle before the next dispatch (no ghosts, no leftover VRAM).

7. **Loop forever** (see NEVER STOP).

### The prompt to give each slot subagent

> You are a research subagent on **slot {i} (GPU {i})**. Work ONLY in
> `worktrees/slot{i}` on branch `autoresearch/<tag>-slot{i}`. Your axis this round is
> **{axis}** and the ONLY ideas you may try are: **{owned_ideas}** (exactly {k}). These are
> **OFF-LIMITS** (other slots own them — never try them): **{off_limits}**. Already tried
> (do not repeat): **{tried_digest}**. The champion is `val_bpb={champion}` — its `train.py`
> is on branch `autoresearch/<tag>`; start from it
> (`git -C worktrees/slot{i} checkout autoresearch/<tag> -- train.py`).
>
> Run **exactly {k} experiment(s), then STOP and return** (do NOT exceed {k} — the
> orchestrator gives your next direction). Before each, run `python3 vast.py status`; if
> under ~9 minutes remain to the deadline, stop early and return what you have. For each:
> 1. Edit `worktrees/slot{i}/train.py` with one of YOUR {k} assigned idea(s) (axis {axis}
>    only). Change only what that idea needs — hold everything else at the champion's values
>    so the delta is attributable.
> 2. `git -C worktrees/slot{i} add -A && git -C worktrees/slot{i} commit -m "<idea>"`.
> 3. Run it **in the foreground and WAIT for it to return** — it blocks ~the training budget
>    then prints the result. **Do not background it, do not poll a log, do not build a wait
>    loop:** `python vast.py exp --slot {i} --train worktrees/slot{i}/train.py`.
> 4. Read the printed `val_bpb` / `peak_vram`. Empty / `CRASH` / `OOM` = it failed. If
>    `OOM`, the idea used too much VRAM — log it `crash`, `git reset --hard HEAD~1`, and
>    DON'T retry it bigger (batch/size is Axis D's job, not yours — just report it).
> 5. Log it to the ledger: `python vast.py log <commit> <val_bpb> <mem_gb>
>    <keep|discard|crash> slot{i} "<description>"` (atomic; the ONLY shared file you write).
> 6. If `val_bpb` improved, keep the commit; else `git -C worktrees/slot{i} reset --hard HEAD~1`.
>
> Do **NOT** edit `findings.md` (orchestrator-only). Stay on your slot/worktree/branch —
> never touch another slot's GPU, worktree, or branch, and never run more than {k}
> experiments. When done, **return a structured summary**: for each experiment — the exact
> one-line diff (so the orchestrator can verify it stayed in axis {axis}), the commit hash,
> `val_bpb` / `peak_vram` (or the failure reason), and keep-or-Tried.

This keeps slots **non-interfering** (distinct GPU + CPU cores + worktree + branch + remote
dir) yet **interconnected** (shared `results.tsv` ledger they write + orchestrator-curated
`findings.md` they read + one champion they all build from).

---

## How an experiment runs (and is bounded)

`vast.py exp` syncs the slot's `train.py`, pins its GPU + CPU cores, runs it, parses the
result, and prints `val_bpb` / `RESULT_JSON`. **Each run is bounded by the training budget
without a brutal kill:** `train.py` self-stops when its elapsed training time hits the
budget (set per session via `vast.py`, default 5 min), so a healthy run ends on its own.
`exp` does **not** force-kill the run; it only reaps a *ghost* still on that slot **before**
launching (clearing a leftover from a died subagent). For a genuinely stuck run, `python
vast.py reap` clears stray runs and shows what's on each GPU.

---

## VRAM & OOM (no OOMs)

Keep peak VRAM well under the GPU's limit (< ~22 GB on a 24 GB 4090). The baseline is sized
to fit; only **Axis D** changes batch/model size. An OOM is a failure: log `crash`,
`git reset --hard`, don't retry larger. Memory-heavy ideas must be paired with a smaller
`DEVICE_BATCH_SIZE` (Axis D's job). `reap` clears stuck runs holding memory.

---

## Logging & objectivity

Log every run via `python vast.py log` (file-locked; concurrency-safe), 6 tab-separated
columns: `commit  val_bpb  memory_gb  status  branch  description` (`0000000` / `0.000000`
/ `0.0` on failure). The fixed budget makes runs comparable; `bench` confirms the GPUs are
equal (spread) and measures the concurrency tax; `vast.py` pins disjoint CPU cores per slot.
When a `val_bpb` delta is small, **re-run on the same slot before crowning a champion**.
Don't `git add` `results.tsv` or `findings.md` (untracked).

---

## Findings (orchestrator-owned)

`findings.md` is the swarm's shared brain and **you are its sole writer.** Subagents return
summaries and log raw runs to `results.tsv`; only you edit `findings.md` (so it never gets
clobbered). Update it every round, right after you reconcile results. Keep **Champion** (val
+ the one-line change), **Tried** (every idea + result), **Dead ends** current. Hand
subagents the latest digest each round.

---

## Teardown

The watchdog auto-destroys the box at the deadline. To stop early: `python vast.py down`,
then `python vast.py ps` to confirm nothing is billing. `extend --hours <H>` pushes the
deadline; `ps`/`nuke` catch and kill orphans; `reap` clears stray runs without destroying
the box.

---

## NEVER STOP

Once the loop begins, don't pause to ask whether to continue. The human may be asleep and
expects research **until the deadline, an interrupt, or teardown**. When a round finishes,
immediately design and dispatch the next — keep all GPUs busy every minute. If ideas run
dry: re-read `train.py`, mine the literature with the `research` skill, combine near-misses
across axes, try more radical changes, rotate axes. The loop runs until stopped, period.
