#!/usr/bin/env python3
"""Self-test for post_stop_watcher.py.

Drives the real watcher code (no reimplementation) against stub ssh / vastai /
pull binaries so that every branch that matters can be exercised in seconds:
the detection path, the verification path, every fail-open path, and the shape of
the real `vastai stop` call -- validated in print-only mode so no instance is ever
touched by the test.

Run:  python3 post_stop_watcher_selftest.py
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WATCHER = HERE / "post_stop_watcher.py"
PY = sys.executable

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# --------------------------------------------------------------------------- #
# Fake checkpoint builder: a structurally real safetensors file with the same
# shape the pull script installs (208 tensors, diffusion_model. prefix).
# --------------------------------------------------------------------------- #

FAKE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def write_fake_ckpt(path: Path, step: int, *, tensors: int = 208,
                    sha: str = FAKE_SHA, prefix: str = "diffusion_model.",
                    keep_default: bool = False, truncate: int = 0) -> None:
    header: dict = {"__metadata__": {"format": "pt", "step": str(step),
                                     "source_sha256": sha,
                                     "remap": "strip .default + prefix diffusion_model."}}
    off = 0
    for i in range(tensors):
        mid = ".default." if keep_default else "."
        name = f"{prefix}blocks.{i}.attn.qkv_proj{mid}lora_A.weight"
        header[name] = {"dtype": "F32", "shape": [2], "data_offsets": [off, off + 8]}
        off += 8
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * (off - truncate))


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

SSH_STUB = r'''#!/usr/bin/env python3
"""Stub ssh. Emits the probe sections from a scenario JSON, or a sha256sum line.

The process sections are emitted in procscan's wire format -- "<pid> <state>"
lines then "OK <live> <undead>" -- because that is what the box now sends. A
scenario can also supply raw lines, which is how the "the count never completed"
case is exercised.
"""
import json, os, sys
sc = json.load(open(os.environ["PSW_TEST_SCENARIO"]))
payload = sys.argv[-1]
if sc.get("ssh_exit"):
    sys.stderr.write("stub: connection refused\n")
    sys.exit(int(sc["ssh_exit"]))
if "sha256sum" in payload:
    print(f'{sc.get("box_sha", "a"*64)}  /workspace/x/step-N.safetensors')
    sys.exit(0)
ck = {int(k): v for k, v in sc["ckpts"].items()}
epoch = sc.get("box_epoch", 1787200000)
target = int(sc.get("target_step", 2000))


def procsec(name, live, undead, raw=None):
    print("###" + name)
    if raw is not None:
        for line in raw:
            print(line)
        return
    for i in range(live):
        print("%d S" % (4000 + i))
    for i in range(undead):
        print("%d Z" % (4900 + i))
    print("OK %d %d" % (live, undead))


print("###NOW"); print(epoch)
print("###SENTINEL")
if sc.get("sentinel"):
    print("stopped_at_cumulative_step=%s" % sc.get("sentinel_step", 2000))
    print("checkpoint=/workspace/x/step-%s.safetensors" % sc.get("sentinel_step", 2000))
    print("do_not_restart=1")
print("###SENTMTIME")
if sc.get("sentinel"):
    print(sc.get("sentinel_mtime", epoch))
print("###HB"); print("ts=x step=%s loss=0.5 attempt=1" % sc.get("hb_step", 661))
print("###HBMTIME"); print(sc.get("hb_mtime", epoch))
print("###CKPT")
for step in sorted(ck):
    print("-rw------- 1 root root %d 1787200000 step-%d.safetensors" % (ck[step], step))
print("-rw------- 1 root root 131227832 1787200000 attempt1_step-600.safetensors")
print("###RECENT"); print(sc.get("recent_writes", 0))
print("###TARGETHDR")
hdr = sc.get("target_hdr", "auto")
if hdr == "auto" and target in ck:
    print("%d %d" % (ck[target], 60000))
elif isinstance(hdr, list):
    print("%d %d" % (hdr[0], hdr[1]))
procsec("TRAINERS", sc.get("trainers", 0), sc.get("trainers_undead", 0),
        sc.get("trainers_raw"))
procsec("SUPERVISORS", sc.get("supervisors", 0), 0, sc.get("supervisors_raw"))
print("###STOPLOG"); print("[ts] STOP_COMPLETE target=2000 stopped_at_step=2000")
print("###END")
'''

VASTAI_STUB = r'''#!/usr/bin/env python3
"""Stub vastai. Records every invocation; flips status to stopped after a stop."""
import json, os, sys
rec = os.environ["PSW_TEST_VASTAI_LOG"]
st = os.environ["PSW_TEST_VASTAI_STATE"]
args = sys.argv[1:]
with open(rec, "a") as fh:
    fh.write(" ".join(args) + "\n")
if "destroy" in args:
    sys.exit(99)  # must never happen
status = open(st).read().strip() if os.path.exists(st) else "running"
if args[:2] == ["stop", "instance"]:
    open(st, "w").write("stopped")
    print("stopping instance %s" % args[2]); sys.exit(0)
if args[:2] == ["show", "instance"]:
    print(json.dumps({"id": int(args[2]), "label": "lobora-h3-a800",
                      "actual_status": status, "cur_state": status,
                      "dph_total": 1.1022}))
    sys.exit(0)
sys.exit(2)
'''

PULL_STUB = r'''#!/usr/bin/env python3
"""Stub pull_latest_lora.py: installs fake-but-structurally-valid checkpoints."""
import json, os, sys
sys.path.insert(0, os.environ["PSW_TEST_HELPER_DIR"])
from ckpt_helper import write_fake_ckpt
from pathlib import Path
sc = json.load(open(os.environ["PSW_TEST_SCENARIO"]))
print("stub pull argv:", sys.argv)
if sc.get("pull_exit"):
    sys.exit(int(sc["pull_exit"]))
dest = Path(sys.argv[sys.argv.index("--dest-dir") + 1])
skip = set(sc.get("pull_skip", []))
bad = sc.get("pull_corrupt", {})
for step in sorted(int(k) for k in sc["ckpts"]):
    if step in skip:
        print("stub: deliberately not installing step-%d" % step); continue
    kw = dict(bad.get(str(step), {}))
    write_fake_ckpt(dest / ("genpt-step-%04d.safetensors" % step), step,
                    sha=sc.get("meta_sha", "a"*64), **kw)
    print("stub: installed step-%d" % step)
'''

def build_env(tmp: Path, scenario: dict, *, print_only: bool = False,
              extra: dict | None = None) -> tuple[dict, Path, Path]:
    helper = tmp / "helper"
    helper.mkdir(parents=True, exist_ok=True)
    # Export the fake-checkpoint builder so the pull stub can use the same code.
    src = Path(__file__).read_text()
    start = src.index("FAKE_SHA =")
    end = src.index("# ---------------------------------------------------------------------------"
                    " #\n# Stubs")
    (helper / "ckpt_helper.py").write_text(
        "import json, struct\nfrom pathlib import Path\n" + src[start:end])

    binz = tmp / "bin"
    binz.mkdir(parents=True, exist_ok=True)
    for name, body in (("ssh", SSH_STUB), ("vastai", VASTAI_STUB), ("pull", PULL_STUB)):
        p = binz / name
        p.write_text(body)
        p.chmod(0o755)

    dest = tmp / "dest"
    dest.mkdir(parents=True, exist_ok=True)
    art = tmp / "art"
    art.mkdir(parents=True, exist_ok=True)
    scen = tmp / "scenario.json"
    scen.write_text(json.dumps(scenario))
    vlog = tmp / "vastai_calls.log"
    vstate = tmp / "vastai_state.txt"
    vstate.write_text("running")

    env = dict(os.environ)
    env.update({
        "PSW_ARTIFACT_DIR": str(art),
        "PSW_LOG": str(art / "post_stop_watcher.log"),
        "PSW_STATE": str(art / "state.json"),
        "PSW_LOCK": str(art / "lock"),
        "PSW_SSH_BIN": str(binz / "ssh"),
        "PSW_VASTAI_BIN": str(binz / "vastai"),
        "PSW_PULL_ARGV": json.dumps([PY, str(binz / "pull")]),
        "PSW_DEST_DIR": str(dest),
        "PSW_POLL_SECS": "1",
        "PSW_CONFIRM_POLLS": "2",
        "PSW_MAX_POLLS": "6",
        "PSW_STOP_POLL_SECS": "1",
        "PSW_STOP_CONFIRM_SECS": "10",
        "PSW_FAIL_SLEEP_SECS": "1",
        "PSW_ALERT_EVERY": "0",
        "PSW_STALL_SECS": "3600",
        # The live cap is 6000 now; these scenarios were written around 2000, so
        # they say so explicitly instead of leaning on the default.
        "PSW_TARGET_STEP": "2000",
        "PSW_TEST_SCENARIO": str(scen),
        "PSW_TEST_VASTAI_LOG": str(vlog),
        "PSW_TEST_VASTAI_STATE": str(vstate),
        "PSW_TEST_HELPER_DIR": str(helper),
        "PSW_PRINT_ONLY": "1" if print_only else "0",
    })
    if extra:
        env.update(extra)
    return env, vlog, dest


def run(env: dict, timeout: int = 180) -> tuple[int, str]:
    proc = subprocess.run([PY, str(WATCHER)], env=env, capture_output=True,
                          text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


FULL = {str(s): 131227832 for s in list(range(100, 2001, 25))}
EARLY = {str(s): 131227832 for s in (100, 200, 300, 400, 500, 600, 625, 650)}


def scenario_fired(**kw) -> dict:
    sc = {"ckpts": FULL, "sentinel": True, "trainers": 0, "supervisors": 0,
          "recent_writes": 0, "hb_step": 2001}
    sc.update(kw)
    return sc


# --------------------------------------------------------------------------- #

def sh_run(script: str) -> str:
    """Run a shell snippet the way ssh runs a remote command: as one argv string,
    so the snippet's own text (including any search pattern in it) is sitting in
    the invoking shell's /proc/<pid>/cmdline. That is the exact shape that made
    the old grep pipeline match itself."""
    proc = subprocess.run(["sh", "-c", script], capture_output=True, text=True,
                          timeout=120)
    return proc.stdout


def t_counter_cannot_match_itself() -> None:
    print("\n[0] the box-side process counter cannot count itself")
    sys.path.insert(0, str(HERE.parent))
    from lobora import procscan

    # The production payload, run locally, for a pattern nothing on this machine
    # can be running. The old `grep -l PAT /proc/[0-9]*/cmdline | wc -l` returned
    # 1 here, every time, because grep was handed its own post-exec argv.
    nothing = "zzzz-no-process-anywhere-has-this-pattern-9876"
    out = sh_run(procscan.remote_scan_command({"TRAINERS": nothing}))
    sec = procscan.parse_section(out.splitlines()[1:])
    check("a pattern matching nothing counts 0, not 1", sec["live"] == 0, out)

    # And the real trainer pattern, which is not running on this machine either.
    real = "model_training/train.py"
    out = sh_run(procscan.remote_scan_command({"TRAINERS": real}))
    sec = procscan.parse_section(out.splitlines()[1:])
    check("the real trainer pattern counts 0 on a box with no trainer",
          sec["live"] == 0, out)

    # For contrast, the construct that was in the file until today.
    old = sh_run('grep -l "model_tr""aining/train.py" /proc/[0-9]*/cmdline '
                 '2>/dev/null | wc -l')
    check("the old grep pipeline really did report a phantom process",
          old.strip() == "1", f"got {old.strip()!r}")

    # A live match is still counted: this must not be zero-happy.
    marker = "lobora-selftest-marker-process"
    child = subprocess.Popen([PY, "-c", "import time; time.sleep(30)", marker],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.4)
        out = sh_run(procscan.remote_scan_command({"TRAINERS": marker}))
        sec = procscan.parse_section(out.splitlines()[1:])
        check("a genuinely running process is still counted",
              sec["live"] == 1 and sec["pids"][0][0] == child.pid,
              f"{sec} pid={child.pid}\n{out}")
    finally:
        child.terminate()
        child.wait(timeout=10)


def t_source_hygiene() -> None:
    print("\n[1] source hygiene")
    src = WATCHER.read_text()
    # "destroy" never exists as a string literal, so it can never reach subprocess.
    check("`destroy` is never a string literal (cannot be passed to vastai)",
          '"destroy"' not in src and "'destroy'" not in src)
    check("no pkill/killall", "pkill" not in src and "killall" not in src)
    check("no process signalling at all", ".kill(" not in src and "os.kill" not in src
          and "signal.SIG" not in src)
    check("no shell=True (so no shell pipeline can hide an exit status)",
          "shell=True" not in src)
    check("stop command is exactly `stop instance`",
          '[VASTAI_BIN, "stop", "instance", VAST_INSTANCE]' in src)
    check("every subprocess status read from returncode",
          src.count("returncode") >= 6)
    # The bug is a /proc cmdline scan whose result is counted through a pipe.
    # Prose about it is fine; a line of code that does it is not.
    offenders = [ln for ln in src.splitlines()
                 if not ln.lstrip().startswith("#")
                 and "cmdline" in ln and "wc -l" in ln]
    check("no grep-over-/proc pipeline survives in the code", not offenders,
          str(offenders))
    check("process counting is delegated to the one shared implementation",
          "procscan.remote_scan_command" in src and "procscan.parse_section" in src)
    check("compiles", subprocess.run([PY, "-m", "py_compile", str(WATCHER)],
                                     capture_output=True).returncode == 0)


def t_waiting() -> None:
    print("\n[2] still training -> no pull, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), {
            "ckpts": EARLY, "sentinel": False, "trainers": 2, "supervisors": 1,
            "hb_step": 661, "recent_writes": 0})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("exits cleanly at MAX_POLLS", rc == 0, f"rc={rc}")
        check("no stop issued", "stop instance" not in calls, calls)
        check("no files installed", not list(dest.glob("*.safetensors")))
        check("logged the wait", "trainers=2" in out)


def t_happy_print_only() -> None:
    print("\n[3] fired, all verified -> stop path in PRINT-ONLY mode")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(), print_only=True)
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("exits 0", rc == 0, f"rc={rc}")
        check("verified every checkpoint", out.count("ok step-") == len(FULL),
              f"{out.count('ok step-')} of {len(FULL)}")
        check("sha cross-check ran", "sha256 cross-check vs box PASSED" in out)
        check("print-only announced the exact real command",
              f"PRINT-ONLY MODE: would run: {env['PSW_VASTAI_BIN']} stop instance "
              f"48056192" in out, out[-1500:])
        check("no actual stop call reached vastai", "stop instance" not in calls, calls)
        check("2000 present locally", (dest / "genpt-step-2000.safetensors").exists())


def t_happy_real_stop() -> None:
    print("\n[4] fired, all verified -> real stop call against stub vastai")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired())
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("exits 0", rc == 0, f"rc={rc}")
        check("issued exactly one stop",
              calls.count("stop instance 48056192") == 1, calls)
        check("never called destroy", "destroy" not in calls, calls)
        check("confirmed stopped state", "CONFIRMED: instance 48056192 reached state "
              "'stopped'" in out, out[-1200:])
        check("state file records the stop",
              json.loads(Path(env["PSW_STATE"]).read_text()).get("stopped_at"))


def t_idempotent() -> None:
    print("\n[5] idempotency -> a second run does not double-stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired())
        run(env)
        first = vlog.read_text().count("stop instance 48056192")
        rc, out = run(env)
        again = vlog.read_text().count("stop instance 48056192")
        check("first run stopped once", first == 1, str(first))
        check("second run issues no new stop", again == first, str(again))
        check("second run said it was already complete",
              "work already complete" in out, out[-800:])
        check("second run exits 0", rc == 0)


def t_pull_fails() -> None:
    print("\n[6] pull failure -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(pull_exit=3))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("alerted", "ALERT [sequence_failed]" in out)
        check("said the instance is left running", "has NOT been stopped" in out)


def t_verify_corrupt() -> None:
    print("\n[7] corrupt step-2000 (wrong tensor count) -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            pull_corrupt={"2000": {"tensors": 12}}))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("named the tensor-count problem",
              "12 tensors, expected 208" in out, out[-1200:])


def t_verify_truncated() -> None:
    print("\n[8] truncated step-2000 -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            pull_corrupt={"2000": {"truncate": 64}}))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("detected truncation", "truncated transfer" in out, out[-1200:])


def t_verify_keys() -> None:
    print("\n[9] un-remapped keys / .default left in -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            pull_corrupt={"1900": {"prefix": "pipe.dit."}}))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop (bad prefix)", "stop instance" not in calls, calls)
        check("named the prefix problem", "key not remapped" in out, out[-1200:])
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            pull_corrupt={"1800": {"keep_default": True}}))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop (.default present)", "stop instance" not in calls, calls)
        check("named the .default problem", "still carries .default" in out, out[-1200:])


def t_missing_file() -> None:
    print("\n[10] a checkpoint silently not installed -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(pull_skip=[2000]))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("named the missing file", "is missing locally" in out, out[-1200:])
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(pull_skip=[1000]))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop when an OLD checkpoint is missing",
              "stop instance" not in calls, calls)


def t_sha_mismatch() -> None:
    print("\n[11] box sha != local metadata sha -> fail-open, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(box_sha=OTHER_SHA))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("named the sha mismatch", "sha256 mismatch" in out, out[-1200:])


def t_fired_without_target() -> None:
    print("\n[12] stopped early with a sentinel but no step-2000 -> preserve, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), {
            "ckpts": EARLY, "sentinel": True, "trainers": 0, "supervisors": 0,
            "recent_writes": 0, "hb_step": 655})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("alerted", "ALERT [fired_without_target]" in out)
        check("still preserved what exists",
              (dest / "genpt-step-0650.safetensors").exists())


def t_dead_early() -> None:
    print("\n[13] run died before 2000, no sentinel -> ALERT + preserve, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), {
            "ckpts": EARLY, "sentinel": False, "trainers": 0, "supervisors": 0,
            "recent_writes": 0, "hb_step": 661})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("alerted dead_early", "ALERT [dead_early]" in out)
        check("said a human should look", "a human should look" in out)
        check("preservation pull ran",
              (dest / "genpt-step-0650.safetensors").exists())
        check("preservation pull ran only once",
              out.count("one-off preservation pull") == 1,
              str(out.count("one-off preservation pull")))


def t_stalled() -> None:
    print("\n[14] heartbeat frozen while trainer alive -> ALERT, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), {
            "ckpts": EARLY, "sentinel": False, "trainers": 2, "supervisors": 1,
            "recent_writes": 0, "hb_step": 661}, extra={"PSW_STALL_SECS": "0"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("alerted stalled", "ALERT [stalled]" in out)
        check("mentioned the wedge", "may be wedged" in out)


def t_ssh_down() -> None:
    print("\n[15] SSH unreachable -> never stops, alerts after 10 failures")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(ssh_exit=255),
                                 extra={"PSW_MAX_POLLS": "12"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("counted consecutive failures", "consecutive SSH failures: 10" in out)
        check("alerted ssh_down", "ALERT [ssh_down]" in out)
        check("explained why it will not stop",
              "cannot be" in out and "verified" in out)


def t_disk_guard() -> None:
    print("\n[16] not enough free disk -> abort before pulling, no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(
            Path(td), scenario_fired(),
            extra={"PSW_DISK_BUFFER_BYTES": str(1 << 60)})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("named the space problem", "not enough free space" in out, out[-900:])
        check("nothing was pulled", not list(dest.glob("*.safetensors")))


def t_lock() -> None:
    print("\n[17] a second concurrent watcher refuses to run")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), {
            "ckpts": EARLY, "sentinel": False, "trainers": 2, "supervisors": 1,
            "recent_writes": 0}, extra={"PSW_MAX_POLLS": "2"})
        Path(env["PSW_LOCK"]).write_text(f"{os.getpid()}\n")
        # Our own pid is alive but is not the watcher's pid, which is exactly the
        # "someone else already holds it" case.
        rc, out = run(env)
        check("second watcher exits 0 without acting", rc == 0, f"rc={rc}")
        check("said another watcher is running", "already running" in out, out[-500:])


def t_quiesce() -> None:
    print("\n[18] checkpoint dir still being written -> waits, does not act")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(recent_writes=3),
                                    extra={"PSW_MAX_POLLS": "5"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("waited for a quiet window", "waiting for a quiet window" in out)
        check("nothing pulled", not list(dest.glob("*.safetensors")))


def t_confirm_polls() -> None:
    print("\n[19] a single stopped-looking read is not enough")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(),
                                    extra={"PSW_MAX_POLLS": "1",
                                           "PSW_CONFIRM_POLLS": "3"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("one poll does not trigger the sequence",
              "stop instance" not in calls and "1/3 confirmations" in out, out[-600:])
        check("nothing pulled on a single read",
              not list(dest.glob("*.safetensors")))


def t_pull_argv() -> None:
    print("\n[20] the pull is invoked with --all and an explicit dest")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(), print_only=True)
        rc, out = run(env)
        logtext = Path(env["PSW_LOG"]).read_text()
        check("--all passed", "'--all'" in logtext, logtext[-600:])
        check("--dest-dir passed", "--dest-dir" in logtext)
        check("pull output captured in the log file",
              "pull_latest_lora.py --all output begins" in logtext)


def t_zombie_does_not_block() -> None:
    print("\n[22] a zombie trainer does not count as alive -> the sequence still runs")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(trainers=0,
                                                             trainers_undead=2))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("reports the undead separately from the living",
              "zombie/dead, not counted" in out, out[:2000])
        check("fires on the normal path anyway",
              "stop condition fired via no-live-trainer" in out, out[-1500:])
        check("issued the stop", calls.count("stop instance 48056192") == 1, calls)
        check("pulled everything first",
              (dest / "genpt-step-2000.safetensors").exists())


def t_override_fires() -> None:
    print("\n[23] a live process lingers but the run is over -> bounded override")
    with tempfile.TemporaryDirectory() as td:
        # Exactly today's evidence: sentinel written, step-2000 complete, nothing
        # written to the ckpt dir, heartbeat frozen for 11000s -- and one trainer
        # process still visible.
        env, vlog, dest = build_env(Path(td), scenario_fired(
            trainers=1, box_epoch=1787211000, hb_mtime=1787200000,
            sentinel_mtime=1787200000))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("exits 0", rc == 0, f"rc={rc}")
        check("announces the override loudly", "ALERT [override]" in out, out[:2500])
        check("says which path fired",
              "stop condition fired via bounded-override" in out, out[-2000:])
        check("quotes the static heartbeat as its evidence",
              "static for 11000s" in out, out[:2500])
        check("still pulled and verified before stopping",
              out.count("ok step-") == len(FULL), str(out.count("ok step-")))
        check("issued exactly one stop",
              calls.count("stop instance 48056192") == 1, calls)


def t_override_waits_for_the_threshold() -> None:
    print("\n[24] the same lingering process, heartbeat static for less -> no override")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(
            trainers=1, box_epoch=1787201000, hb_mtime=1787200000,
            sentinel_mtime=1787200000))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("no override", "ALERT [override]" not in out)
        check("nothing was pulled", not list(dest.glob("*.safetensors")))
        check("counts down to the override instead of going quiet",
              "bounded override needs 5400s of static heartbeat and has 1000s" in out
              and "4400s to go" in out, out[-1500:])


def t_override_still_fails_open() -> None:
    print("\n[25] the override fires but the pull fails -> still no stop")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            trainers=1, box_epoch=1787211000, hb_mtime=1787200000,
            sentinel_mtime=1787200000, pull_exit=3))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("the override did fire", "ALERT [override]" in out, out[:2000])
        check("but no stop was issued", "stop instance" not in calls, calls)
        check("said the instance is left running", "has NOT been stopped" in out)
    print("     ...and the same when a checkpoint fails verification")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, _ = build_env(Path(td), scenario_fired(
            trainers=1, box_epoch=1787211000, hb_mtime=1787200000,
            sentinel_mtime=1787200000, pull_corrupt={"2000": {"truncate": 64}}))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("named the truncation", "truncated transfer" in out, out[-1200:])


def t_override_needs_a_complete_target() -> None:
    print("\n[26] lingering process + a half-written target -> no override")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(
            trainers=1, box_epoch=1787211000, hb_mtime=1787200000,
            sentinel_mtime=1787200000,
            target_hdr=[131227832, 900000000]))   # header longer than the file
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("no override", "ALERT [override]" not in out)
        check("named the truncated-looking target",
              "truncated or still being written" in out, out[-1500:])


def t_stale_sentinel_from_a_previous_run() -> None:
    print("\n[27] relaunched run: last run's sentinel and step-2000 must not fire")
    with tempfile.TemporaryDirectory() as td:
        # The live situation: cap is now 6000, the box still holds the old
        # sentinel (step 2000) and every checkpoint up to 2050 from the new run.
        ck = {str(s): 131227832 for s in list(range(100, 2051, 25))}
        env, vlog, dest = build_env(Path(td), {
            "ckpts": ck, "sentinel": True, "sentinel_step": 2000,
            "sentinel_mtime": 1787200000, "hb_mtime": 1787209000,
            "box_epoch": 1787209000, "hb_step": 2051, "trainers": 3,
            "supervisors": 1, "recent_writes": 0, "target_step": 6000},
            extra={"PSW_TARGET_STEP": "6000"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("marks the sentinel stale", "sentinel=stale" in out, out[-1500:])
        check("says why: it records a step below the new cap",
              "below the configured target 6000" in out, out[-1500:])
        check("pulled nothing", not list(dest.glob("*.safetensors")))
    print("     ...and a sentinel AT the new cap is still stale if training ran after")
    with tempfile.TemporaryDirectory() as td:
        ck = {str(s): 131227832 for s in list(range(100, 6001, 25))}
        env, vlog, dest = build_env(Path(td), {
            "ckpts": ck, "sentinel": True, "sentinel_step": 6000,
            "sentinel_mtime": 1787200000, "hb_mtime": 1787209000,
            "box_epoch": 1787209000, "hb_step": 6051, "trainers": 3,
            "supervisors": 1, "recent_writes": 0, "target_step": 6000},
            extra={"PSW_TARGET_STEP": "6000"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("says why: the heartbeat outlived the sentinel",
              "AFTER the sentinel" in out, out[-1500:])
    print("     ...and not even when the trainer briefly disappears mid-relaunch")
    with tempfile.TemporaryDirectory() as td:
        ck = {str(s): 131227832 for s in list(range(100, 2051, 25))}
        env, vlog, dest = build_env(Path(td), {
            "ckpts": ck, "sentinel": True, "sentinel_step": 2000,
            "sentinel_mtime": 1787200000, "hb_mtime": 1787209000,
            "box_epoch": 1787209000, "hb_step": 2051, "trainers": 0,
            "supervisors": 1, "recent_writes": 0, "target_step": 6000},
            extra={"PSW_TARGET_STEP": "6000"})
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("still no stop", "stop instance" not in calls, calls)
        check("treats it as a run that died early, not a finished one",
              "ALERT [dead_early]" in out, out[-1500:])


def t_count_unknown_never_reads_as_zero() -> None:
    print("\n[28] the box-side count did not complete -> UNKNOWN, never 0")
    with tempfile.TemporaryDirectory() as td:
        env, vlog, dest = build_env(Path(td), scenario_fired(
            trainers_raw=["/bin/sh: 1: python3: not found"]))
        rc, out = run(env)
        calls = vlog.read_text() if vlog.exists() else ""
        check("no stop issued", "stop instance" not in calls, calls)
        check("logs the count as UNKNOWN", "trainers=UNKNOWN" in out, out[-1500:])
        check("alerts about it", "ALERT [count_unknown]" in out, out[-1500:])
        check("pulled nothing", not list(dest.glob("*.safetensors")))


def t_real_pull_script_flags() -> None:
    print("\n[21] the real pull script accepts the flags we pass")
    real = HERE / "pull_latest_lora.py"
    if not real.exists():
        check("real pull script present", False, str(real))
        return
    proc = subprocess.run([PY, str(real), "--help"], capture_output=True, text=True,
                          timeout=60)
    check("real --help works", proc.returncode == 0, proc.stderr[-300:])
    check("real script has --all", "--all" in proc.stdout)
    check("real script has --dest-dir", "--dest-dir" in proc.stdout)
    # Deliberately no git command here: another agent owns that repo's index. Import
    # the watcher with a scrubbed environment and read the paths it would write to,
    # rather than grepping for a checkout path that differs per machine.
    repo = HERE.parent
    probe = (
        "import importlib.util;"
        f"spec=importlib.util.spec_from_file_location('psw', r'{WATCHER}');"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print(m.ARTIFACT_DIR);print(m.LOG_PATH);print(m.STATE_PATH);print(m.LOCK_PATH)"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PSW_")}
    got = subprocess.run([PY, "-c", probe], env=clean, capture_output=True,
                         text=True, timeout=60)
    paths = [Path(p) for p in got.stdout.split()]
    inside = [str(p) for p in paths if p == repo or repo in p.parents]
    check("the watcher's default log/state/lock live outside the repo",
          len(paths) == 4 and not inside,
          f"{inside} {got.stderr[-200:]}")


def main() -> int:
    print(f"post_stop_watcher self-test  ({WATCHER})")
    for fn in (t_counter_cannot_match_itself,
               t_source_hygiene, t_waiting, t_happy_print_only, t_happy_real_stop,
               t_idempotent, t_pull_fails, t_verify_corrupt, t_verify_truncated,
               t_verify_keys, t_missing_file, t_sha_mismatch, t_fired_without_target,
               t_dead_early, t_stalled, t_ssh_down, t_disk_guard, t_lock, t_quiesce,
               t_confirm_polls, t_pull_argv,
               t_zombie_does_not_block, t_override_fires,
               t_override_waits_for_the_threshold, t_override_still_fails_open,
               t_override_needs_a_complete_target,
               t_stale_sentinel_from_a_previous_run,
               t_count_unknown_never_reads_as_zero,
               t_real_pull_script_flags):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            global FAIL
            FAIL += 1
            print(f"  FAIL  {fn.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
