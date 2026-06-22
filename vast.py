#!/usr/bin/env python3
"""Vast.ai control plane for autoresearch — one multi-GPU box per session.

The research loop and the AI agents run on YOUR laptop. The only thing that goes
to Vast is the GPU work: each experiment is `train.py` running for the fixed
5-minute budget. Because `train.py` honors CUDA_VISIBLE_DEVICES, N experiments can
run truly in parallel on one N-GPU box — each pinned to its own GPU (separate
VRAM => no interference) and its own slice of CPU cores via taskset (the dataloader
tokenizes on the CPU, so disjoint cores keep concurrent runs from stealing each
other's throughput). `bench` measures any remaining contention so results stay
objective.

Safety model (same spirit as the battle-tested arc script):
  * Exactly ONE tracked box at a time, recorded in .vast_state.json with its price
    and a hard deadline. `watchdog` auto-destroys at the deadline. `status` shows
    cost so far. `down` tears it down; `ps`/`nuke` catch orphans across the account.
  * `up` refuses offers above the per-GPU price cap and always prints the bill rate.

Auth: run `vastai set api-key <KEY>` once (stored locally). An ssh key is
auto-generated + registered if missing.

Typical session (what program.md drives the agent to do):
  vast.py up --gpus 4 --hours 3        # rent cheapest qualifying 4-GPU box
  vast.py watchdog &                   # background auto-destroy at the deadline
  vast.py setup                        # uv sync + prepare.py + detect GPUs/CPUs
  vast.py bench                        # confirm GPUs equivalent + measure contention
  vast.py exp --slot 0 --train worktrees/slot0/train.py   # one experiment on GPU 0
  ...                                  # subagents run many of these in parallel
  vast.py down                         # destroy + clear state (or let watchdog do it)

Commands:
  search   list cheapest qualifying offers (per-GPU price)
  up       rent cheapest qualifying box + record price/deadline
  status   instance status, GPUs/CPUs, uptime, est. cost, deadline
  sync     upload the repo to the box (tar over ssh)
  setup    sync + install uv + uv sync + prepare.py + detect GPU/CPU topology
  bench    solo + concurrent throughput per GPU (objectivity check)
  exp      --slot N --train PATH: run that train.py on GPU N, print val_bpb/vram
  run      run an arbitrary command in the remote repo dir
  log      atomically append a row to local results.tsv (for parallel subagents)
  pull     download a file/dir back from the box
  watchdog loop; destroy at the recorded deadline (run in background)
  extend   push the deadline out
  down     destroy the tracked box + clear state
  ps       list ALL instances on the account (catch orphans)
  nuke     destroy ALL instances on the account
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
STATE = REPO / ".vast_state.json"
REMOTE_DIR = "/root/auto"
RESULTS = REPO / "results.tsv"

# Use the proven "PyTorch (Vast)" template purely for a fast-booting image that has
# CUDA + python + a working sshd. We do NOT use its torch — `uv sync` builds a fresh
# .venv with the cu128 torch this repo pins. Override with --image if you prefer.
TEMPLATE_HASH = "a33b72bd045341cfcd678ce7c932a614"  # PyTorch (Vast)
DEFAULT_GPU = "RTX_4090"

# Offer filter (applied in Python on the raw JSON, robust to query syntax).
MIN_CUDA = 12.8          # cu128 torch needs a host driver supporting CUDA 12.8
MIN_RELIABILITY = 0.98
MIN_INET_DOWN = 300      # Mbit/s — HF data + torch wheels download fast
MIN_DLPERF = 110         # avoid throttled/junk hosts (a real 4090 is ~150)
MIN_CPU_PER_GPU = 6      # the dataloader tokenizes on CPU; give each slot real cores
MIN_RAM_PER_GPU = 16     # GB
BLOCK_COUNTRIES = ("CN",)  # HuggingFace is often blocked/slow here


def _vastai() -> str:
    return shutil.which("vastai") or str(Path.home() / ".local/bin/vastai")


def vast(args: list[str], raw: bool = False, check: bool = True):
    cmd = [_vastai(), *args]
    if raw:
        cmd.append("--raw")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{p.stderr or p.stdout}")
    if raw:
        try:
            return json.loads(p.stdout or "null")
        except json.JSONDecodeError:
            sys.exit(f"could not parse JSON from: vastai {' '.join(args)}\n{p.stdout}")
    return p.stdout


def load_state() -> dict | None:
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save_state(d: dict) -> None:
    STATE.write_text(json.dumps(d, indent=2))


def require_state() -> dict:
    s = load_state()
    if not s:
        sys.exit("no tracked instance (.vast_state.json). Run: vast.py up")
    return s


# --- ssh -----------------------------------------------------------------------

SSH_KEY = Path.home() / ".ssh/id_ed25519"


def ensure_ssh_key() -> None:
    pub = SSH_KEY.with_suffix(".pub")
    if not pub.exists():
        print("generating an ssh key…")
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(SSH_KEY)], check=True)
    subprocess.run([_vastai(), "create", "ssh-key", pub.read_text().strip()],
                   capture_output=True, text=True)  # idempotent


def _ssh_opts() -> list[str]:
    opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR"]
    if SSH_KEY.exists():
        opts += ["-i", str(SSH_KEY)]
    return opts


def ssh_base(s: dict) -> list[str]:
    return ["ssh", *_ssh_opts(), "-p", str(s["port"]), f"root@{s['host']}"]


def ssh_run(s: dict, remote_cmd: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_base(s) + [remote_cmd], capture_output=True, text=True, check=check)


def scp_to(s: dict, local: str, remote: str) -> None:
    subprocess.run(["scp", *_ssh_opts(), "-P", str(s["port"]), local,
                    f"root@{s['host']}:{remote}"], check=True)


def attach_ssh(s: dict) -> None:
    pub = SSH_KEY.with_suffix(".pub")
    if pub.exists():
        subprocess.run([_vastai(), "attach", "ssh", str(s["instance_id"]), pub.read_text().strip()],
                       capture_output=True, text=True)


def wait_ssh(s: dict, tries: int = 36) -> bool:
    r = None
    for _ in range(tries):
        r = ssh_run(s, "true")
        if r.returncode == 0:
            return True
        time.sleep(5)
    print("WARN: ssh not ready:", (r.stderr if r else "").strip()[:200])
    return False


def _ssh_endpoint(iid) -> tuple[str | None, int | None]:
    out = vast(["ssh-url", str(iid)], check=False) or ""
    m = re.match(r"ssh://[^@]+@([^:\s]+):(\d+)", out.strip())
    return (m.group(1), int(m.group(2))) if m else (None, None)


# --- offers --------------------------------------------------------------------

def find_offers(gpu: str, gpus: int, max_price_per_gpu: float) -> list[dict]:
    # verified=true keeps Vast's vetted hosts; num_gpus pins a whole N-GPU machine
    # (all GPUs identical => cross-slot comparability).
    offers = vast(["search", "offers",
                   f"gpu_name={gpu} num_gpus={gpus} rentable=true verified=true",
                   "-o", "dph_total"], raw=True) or []
    machine_cap = max_price_per_gpu * gpus
    ok = []
    for o in offers:
        country = str(o.get("geolocation", "")).split(",")[-1].strip()
        cpu = o.get("cpu_cores_effective") or o.get("cpu_cores", 0)
        ram = o.get("cpu_ram", 0) / 1024  # MB -> GB (total box RAM)
        if (o.get("cuda_max_good", 0) >= MIN_CUDA
                and o.get("reliability2", 0) >= MIN_RELIABILITY
                and o.get("inet_down", 0) >= MIN_INET_DOWN
                and o.get("dlperf", 0) >= MIN_DLPERF
                and cpu >= MIN_CPU_PER_GPU * gpus
                and ram >= MIN_RAM_PER_GPU * gpus
                and o.get("dph_total", 1e9) <= machine_cap
                and country not in BLOCK_COUNTRIES):
            ok.append(o)
    ok.sort(key=lambda o: o["dph_total"])
    return ok


def fmt_offer(o: dict) -> str:
    n = o.get("num_gpus", 1)
    cpu = o.get("cpu_cores_effective") or o.get("cpu_cores", 0)
    return (f"id={o['id']:<10} ${o['dph_total']:.3f}/hr (${o['dph_total']/n:.3f}/gpu)  "
            f"{n}x {o['gpu_name']}  cpu={cpu:.0f}  dlperf={o.get('dlperf', 0):.0f}  "
            f"cuda={o.get('cuda_max_good')}  inet={o.get('inet_down', 0):.0f}↓  "
            f"{o.get('geolocation', '?')}")


# --- topology / slots ----------------------------------------------------------

def slot_cores(s: dict, slot: int) -> tuple[int, str]:
    """Return (cores_per_slot, taskset_range) for a slot, e.g. (8, '8-15')."""
    cps = max(1, s.get("cpu_cores", s["num_gpus"]) // s["num_gpus"])
    lo = slot * cps
    return cps, f"{lo}-{lo + cps - 1}"


# --- commands ------------------------------------------------------------------

def cmd_search(a) -> None:
    offers = find_offers(a.gpu, a.gpus, a.max_price)
    if not offers:
        print(f"no {a.gpus}x {a.gpu} offers <= ${a.max_price}/gpu/hr meeting filters")
        return
    print(f"top {min(8, len(offers))} cheapest qualifying {a.gpus}x {a.gpu} offers:")
    for o in offers[:8]:
        print("  " + fmt_offer(o))


def cmd_up(a) -> None:
    if load_state():
        sys.exit("an instance is already tracked. `status`/`down` first (or rm .vast_state.json).")
    ensure_ssh_key()
    offers = find_offers(a.gpu, a.gpus, a.max_price)
    if not offers:
        sys.exit(f"no {a.gpus}x {a.gpu} offers <= ${a.max_price}/gpu/hr meeting filters "
                 f"(try --max-price, fewer --gpus, or a different --gpu)")
    if a.offer_id:
        best = next((o for o in offers if o["id"] == a.offer_id), None)
        if not best:
            sys.exit(f"offer {a.offer_id} not in qualifying set; run `search`")
    else:
        best = offers[0]
    print("renting: " + fmt_offer(best))
    create = ["create", "instance", str(best["id"]), "--disk", str(a.disk)]
    create += ["--image", a.image] if a.image else ["--template_hash", TEMPLATE_HASH]
    res = vast(create, raw=True)
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        sys.exit(f"create failed: {res}")
    now = time.time()
    save_state({
        "instance_id": iid, "offer_id": best["id"], "dph": best["dph_total"],
        "gpu": best["gpu_name"], "num_gpus": best.get("num_gpus", a.gpus),
        "cpu_cores": int(best.get("cpu_cores_effective") or best.get("cpu_cores", 0)),
        "created_at": now, "deadline": now + a.hours * 3600,
        "host": None, "port": None, "ready": False,
    })
    print(f"created instance {iid} at ${best['dph_total']:.3f}/hr "
          f"({best.get('num_gpus', a.gpus)} GPUs); auto-destroy deadline in {a.hours}h.")
    print("IMPORTANT: start the watchdog now so the box can't be left billing:\n"
          "  python vast.py watchdog &")
    print("waiting for it to boot…")
    _wait_running()


def _wait_running() -> None:
    s = require_state()
    for _ in range(60):
        info = vast(["show", "instance", str(s["instance_id"])], raw=True)
        status = info.get("actual_status")
        print(f"  status={status}")
        if status == "running":
            host, port = _ssh_endpoint(s["instance_id"])
            if host:
                s["host"], s["port"] = host, port
                save_state(s)
                print(f"READY. ssh -p {port} root@{host}\nnext: python vast.py setup")
                return
        time.sleep(10)
    print("still not running after 10 min; check `vast.py status` / the Vast UI")


def cmd_status(a) -> None:
    s = load_state()
    if not s:
        print("no tracked instance.")
        return
    info = vast(["show", "instance", str(s["instance_id"])], raw=True, check=False)
    up_h = (time.time() - s["created_at"]) / 3600
    print(f"instance {s['instance_id']}  {s['num_gpus']}x {s['gpu']}  "
          f"${s['dph']:.3f}/hr  cpu_cores={s.get('cpu_cores', '?')}")
    print(f"  status: {info.get('actual_status') if isinstance(info, dict) else '?'}")
    print(f"  uptime: {up_h:.2f}h   est. cost so far: ${up_h * s['dph']:.2f}")
    print(f"  deadline in: {(s['deadline'] - time.time()) / 3600:.2f}h")
    if s.get("host"):
        print(f"  ssh -p {s['port']} root@{s['host']}")
        print(f"  slots: " + ", ".join(f"slot{i}->gpu{i} (cores {slot_cores(s, i)[1]})"
                                        for i in range(s["num_gpus"])))


def _resolve_endpoint(s: dict) -> dict:
    host, port = _ssh_endpoint(s["instance_id"])
    if host:
        s["host"], s["port"] = host, port
        save_state(s)
    elif not s.get("host"):
        _wait_running()
        s = require_state()
    return s


def cmd_sync(a) -> None:
    s = _resolve_endpoint(require_state())
    attach_ssh(s)
    if not wait_ssh(s):
        sys.exit("ssh never came up; check `vast.py status` / the Vast UI")
    print(f"uploading repo -> {REMOTE_DIR} (tar over ssh)…")
    excludes = " ".join(f"--exclude=./{x}" for x in
                        (".venv", ".git", "worktrees", ".vast_state.json", "results.tsv",
                         "__pycache__", "dev", "queue", "results"))
    remote = " ".join(ssh_base(s)) + f" 'mkdir -p {REMOTE_DIR} && tar xzf - -C {REMOTE_DIR}'"
    subprocess.run(f"tar czf - {excludes} -C {REPO} . | {remote}", shell=True, check=True)
    print("upload complete.")


SETUP_SCRIPT = r"""
set -e
export PATH="$HOME/.local/bin:/venv/main/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd {REMOTE_DIR}

echo "=== ensure uv ==="
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv 2>/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== uv sync (builds .venv with the repo's cu128 torch) ==="
uv sync

echo "=== prepare.py (download shards + train tokenizer; idempotent) ==="
uv run prepare.py --num-shards {NUM_SHARDS}

echo "=== topology ==="
echo "GPU_COUNT=$(nvidia-smi -L | wc -l)"
echo "CPU_COUNT=$(nproc)"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== setup complete ==="
"""


def cmd_setup(a) -> None:
    cmd_sync(a)
    s = require_state()
    script = SETUP_SCRIPT.format(REMOTE_DIR=REMOTE_DIR, NUM_SHARDS=a.num_shards)
    # Stream setup output live so the user/agent can watch the (slow) data prep.
    p = subprocess.run(ssh_base(s) + [script])
    if p.returncode != 0:
        sys.exit("setup failed; see output above")
    # Re-detect real GPU/CPU counts from the box (offer metadata can be approximate).
    out = ssh_run(s, "echo GPU=$(nvidia-smi -L | wc -l) CPU=$(nproc)").stdout
    gm = re.search(r"GPU=(\d+)", out)
    cm = re.search(r"CPU=(\d+)", out)
    if gm:
        s["num_gpus"] = int(gm.group(1))
    if cm:
        s["cpu_cores"] = int(cm.group(1))
    s["ready"] = True
    save_state(s)
    print(f"\ntopology: {s['num_gpus']} GPUs, {s['cpu_cores']} CPU cores "
          f"=> {slot_cores(s, 0)[0]} cores/slot")
    print("slots:", ", ".join(f"slot{i}->gpu{i} (cores {slot_cores(s, i)[1]})"
                              for i in range(s["num_gpus"])))
    print("next: python vast.py bench   (confirm GPUs are comparable)")


# --- experiment run ------------------------------------------------------------

METRIC_KEYS = ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
               "mfu_percent", "total_tokens_M", "num_steps", "num_params_M", "depth")


def parse_metrics(text: str) -> dict:
    out = {}
    for k in METRIC_KEYS:
        m = re.search(rf"^{k}:\s*([\d.]+)\s*$", text, re.MULTILINE)
        if m:
            out[k] = float(m.group(1))
    return out


def _train_invocation(s: dict, slot: int, script: str, cache_dir: str | None = None) -> str:
    """env -> taskset -> python <script> as one exec chain (no shell, no cd).

    No subshell means `timeout` can signal the python process directly and free the
    GPU. PYTHONPATH lets train.py import prepare.py regardless of cwd. CUDA_VISIBLE_
    DEVICES pins the slot's GPU; taskset + OMP cap it to disjoint CPU cores so the
    on-CPU tokenizer in concurrent runs doesn't steal this slot's throughput. A
    persistent TORCHINDUCTOR_CACHE_DIR makes torch.compile reuse artifacts across
    runs (warm starts → less startup overhead, and a fast bench)."""
    cps, cores = slot_cores(s, slot)
    gpu = slot  # slot N -> GPU N
    cache = f"TORCHINDUCTOR_CACHE_DIR={cache_dir} " if cache_dir else ""
    return (f"taskset -c {cores} env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS={cps} "
            f"OPENBLAS_NUM_THREADS={cps} PYTHONPATH={REMOTE_DIR} {cache}"
            f"{REMOTE_DIR}/.venv/bin/python {script}")


def cmd_exp(a) -> None:
    """Run ONE experiment: push this slot's train.py to the box, run it on the
    slot's GPU (CPU-pinned), and report val_bpb / peak_vram. Used by subagents."""
    s = require_state()
    if not s.get("ready"):
        print("WARN: box not marked ready; run `vast.py setup` first.", file=sys.stderr)
    slot = a.slot
    if slot >= s["num_gpus"]:
        sys.exit(f"slot {slot} out of range (box has {s['num_gpus']} GPUs)")
    train = Path(a.train)
    if not train.exists():
        sys.exit(f"train file not found: {train}")
    workdir = f"{REMOTE_DIR}/slots/slot{slot}"
    cache = f"{REMOTE_DIR}/.inductor/slot{slot}"  # per-slot compile cache (warm starts)
    ssh_run(s, f"mkdir -p {workdir} {cache}")
    scp_to(s, str(train), f"{workdir}/train.py")
    log = f"{workdir}/run.log"
    remote = (_train_invocation(s, slot, f"{workdir}/train.py", cache) + f" > {log} 2>&1; "
              f"grep -E '^({'|'.join(METRIC_KEYS)}):' {log} || "
              f"(echo '--- CRASH (tail) ---'; tail -n 40 {log})")
    print(f"[slot{slot}/gpu{slot}] running train.py (cores {slot_cores(s, slot)[1]})…",
          flush=True)
    t0 = time.time()
    r = ssh_run(s, remote)
    dt = time.time() - t0
    m = parse_metrics(r.stdout)
    if "val_bpb" in m:
        result = {"slot": slot, "ok": True, "wall_seconds": round(dt, 1), **m}
        print(f"[slot{slot}] val_bpb={m['val_bpb']:.6f}  "
              f"vram={m.get('peak_vram_mb', 0)/1024:.1f}GB  "
              f"steps={int(m.get('num_steps', 0))}  wall={dt:.0f}s")
    else:
        result = {"slot": slot, "ok": False, "wall_seconds": round(dt, 1)}
        print(f"[slot{slot}] CRASH / no val_bpb. tail:\n{r.stdout[-1500:]}")
    print("RESULT_JSON:" + json.dumps(result))
    sys.exit(0 if result["ok"] else 1)


# --- bench (objectivity) -------------------------------------------------------

def _tokps_series(text: str) -> list[int]:
    return [int(x.replace(",", "")) for x in re.findall(r"tok/sec:\s*([\d,]+)", text)]


def _steady_tokps(text: str) -> float | None:
    """Median tok/sec over the back half of the run (drops compile/warmup ramp)."""
    s = _tokps_series(text)
    if len(s) < 4:
        return float(statistics.median(s)) if s else None
    return float(statistics.median(s[len(s) // 2:]))


BENCH_CACHE = f"{REMOTE_DIR}/.inductor/bench"  # shared so phase B starts warm


def _bench_one(s: dict, slot: int, seconds: int) -> float | None:
    # Run the BASELINE train.py (already synced to REMOTE_DIR) for `seconds`, then
    # kill it and read the streamed tok/sec. Faithful: real model + real dataloader.
    # `timeout -k` exec-signals the python directly so the GPU is freed promptly.
    cmd = (f"mkdir -p {BENCH_CACHE}; timeout -k 10 -s TERM {seconds} "
           + _train_invocation(s, slot, f"{REMOTE_DIR}/train.py", BENCH_CACHE))
    return _steady_tokps(ssh_run(s, cmd).stdout)


def cmd_bench(a) -> None:
    """~1 minute objectivity check. Phase A warms the shared compile cache on GPU 0
    (and gives the solo throughput); Phase B then measures all GPUs at once with a
    warm cache, so a short window is enough. GPU equivalence = spread across the
    concurrent readings; contention = solo vs concurrent."""
    s = require_state()
    if not s.get("ready"):
        sys.exit("run `vast.py setup` first")
    G = s["num_gpus"]
    warm = a.seconds + 25  # cold run: pay torch.compile once + leave a steady window
    print(f"bench (~1 min): GPU0 solo+warmup ({warm}s), then {G} GPUs concurrent ({a.seconds}s).")

    print("— solo / warm-up (gpu0) —")
    solo = _bench_one(s, 0, warm)
    print(f"  gpu0: {solo:,.0f} tok/sec" if solo else "  gpu0: no reading")

    report = {"seconds": a.seconds, "solo_gpu0": solo}
    if G == 1:
        report["note"] = "single-GPU box: no parallelism, nothing to contend"
        (REPO / "bench.json").write_text(json.dumps(report, indent=2))
        print("\nsingle-GPU box — comparisons are trivially objective. saved bench.json.")
        return

    print(f"— concurrent (all {G} GPUs at once, warm cache) —")
    with ThreadPoolExecutor(max_workers=G) as ex:
        conc = dict(zip(range(G), ex.map(lambda i: _bench_one(s, i, a.seconds), range(G))))
    for i in range(G):
        print(f"  gpu{i}: {conc[i]:,.0f} tok/sec" if conc[i] else f"  gpu{i}: no reading")

    cv = [v for v in conc.values() if v]
    report["concurrent"] = conc
    if cv:
        spread = (max(cv) - min(cv)) / statistics.median(cv) * 100
        report["gpu_spread_pct"] = round(spread, 1)
        verdict = "EQUIVALENT" if spread <= 5 else "UNEVEN — treat cross-GPU results with care"
        print(f"\nGPU spread (under load): {spread:.1f}%  -> {verdict}")
    if solo and cv:
        tax = (1 - statistics.median(cv) / solo) * 100
        report["contention_tax_pct"] = round(tax, 1)
        if tax <= 5:
            note = "negligible — full parallel is objective"
        elif tax <= 15:
            note = "modest — fine, but compare experiments run at the SAME parallelism"
        else:
            note = "HIGH — consider fewer parallel slots for clean comparisons"
        print(f"Concurrency tax ({G}-up vs solo): {tax:.1f}%  -> {note}")
    (REPO / "bench.json").write_text(json.dumps(report, indent=2))
    print("\nsaved bench.json (read this before trusting cross-run comparisons).")


# --- misc commands -------------------------------------------------------------

def cmd_run(a) -> None:
    s = require_state()
    sys.exit(subprocess.run(ssh_base(s) + [f"cd {REMOTE_DIR} && {a.command}"]).returncode)


def cmd_pull(a) -> None:
    s = require_state()
    src = f"root@{s['host']}:{REMOTE_DIR}/{a.remote}"
    subprocess.run(["scp", *_ssh_opts(), "-P", str(s["port"]), "-r", src, a.local], check=True)


def cmd_log(a) -> None:
    """Atomically append one tab-separated row to results.tsv (parallel-safe).
    Subagents call this so concurrent writes never interleave/corrupt the ledger."""
    import fcntl
    row = "\t".join(a.fields)
    if not RESULTS.exists():
        RESULTS.write_text("commit\tval_bpb\tmemory_gb\tstatus\tbranch\tdescription\n")
    with open(RESULTS, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print("logged:", row)


def cmd_watchdog(a) -> None:
    print("watchdog: will auto-destroy at the recorded deadline; Ctrl-C to stop.")
    while True:
        s = load_state()
        if not s:
            print("watchdog: no tracked instance; exiting.")
            return
        if time.time() >= s["deadline"]:
            print("watchdog: deadline reached -> destroying instance.")
            _destroy(s["instance_id"])
            STATE.unlink(missing_ok=True)
            return
        time.sleep(60)


def _destroy(iid) -> None:
    subprocess.run([_vastai(), "destroy", "instance", str(iid)], input="y\n", text=True)


def cmd_extend(a) -> None:
    s = require_state()
    s["deadline"] = time.time() + a.hours * 3600
    save_state(s)
    print(f"deadline extended to {a.hours}h from now")


def cmd_down(a) -> None:
    s = require_state()
    _destroy(s["instance_id"])
    STATE.unlink(missing_ok=True)
    print(f"destroyed {s['instance_id']} and cleared state.")


def cmd_ps(a) -> None:
    rows = vast(["show", "instances"], raw=True) or []
    if not rows:
        print("no instances on the account. (nothing being billed)")
        return
    print("ALL instances on the account:")
    for r in rows:
        print(f"  id={r.get('id')}  {r.get('num_gpus')}x {r.get('gpu_name')}  "
              f"{r.get('actual_status')}  ${r.get('dph_total', 0):.3f}/hr")


def cmd_nuke(a) -> None:
    rows = vast(["show", "instances"], raw=True) or []
    if not rows:
        print("nothing to nuke.")
        return
    for r in rows:
        print(f"destroying {r.get('id')} ({r.get('gpu_name')})")
        _destroy(r.get("id"))
    STATE.unlink(missing_ok=True)
    print("all instances destroyed.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        return p

    for name in ("search", "up"):
        p = add(name, cmd_search if name == "search" else cmd_up)
        p.add_argument("--gpu", default=DEFAULT_GPU)
        p.add_argument("--gpus", type=int, default=4, help="GPUs on the box = max parallel slots")
        p.add_argument("--max-price", type=float, default=0.60, help="cap in $/GPU/hr")
        if name == "up":
            p.add_argument("--hours", type=float, default=3.0, help="session length (auto-destroy)")
            p.add_argument("--disk", type=int, default=80)
            p.add_argument("--image", default=None, help="raw docker image (default: Vast PyTorch template)")
            p.add_argument("--offer-id", type=int, default=None, help="rent this exact offer id")

    add("status", cmd_status)
    add("sync", cmd_sync)
    sp = add("setup", cmd_setup)
    sp.add_argument("--num-shards", type=int, default=16, help="train shards to download")
    bp = add("bench", cmd_bench)
    bp.add_argument("--seconds", type=int, default=15,
                    help="warm measurement window per phase (total bench ~1 min)")
    ep = add("exp", cmd_exp)
    ep.add_argument("--slot", type=int, required=True)
    ep.add_argument("--train", required=True, help="path to the train.py to run")
    rp = add("run", cmd_run)
    rp.add_argument("command")
    pp = add("pull", cmd_pull)
    pp.add_argument("remote")
    pp.add_argument("local")
    lp = add("log", cmd_log)
    lp.add_argument("fields", nargs="+", help="commit val_bpb memory_gb status branch description")
    add("watchdog", cmd_watchdog)
    xp = add("extend", cmd_extend)
    xp.add_argument("--hours", type=float, default=2.0)
    add("down", cmd_down)
    add("ps", cmd_ps)
    add("nuke", cmd_nuke)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
