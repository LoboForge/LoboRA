#!/usr/bin/env python3
"""Self-test for lora_pull_watcher.py.

Drives the real watcher code (no reimplementation) against stub ssh / vastai / pull
binaries, so the branches that decide whether a pull happens can be exercised in
seconds: the no-op cycle, the pull cycle, the disk guard, transient and permanent
SSH failure, the handoff to post_stop_watcher, the failsafe when post_stop_watcher
is dead, the mid-pull yield, and the lock (live, and stale from a dead owner).

Nothing here touches the real box, the real ComfyUI directory, or vastai.

Run:  python3 lora_pull_watcher_selftest.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WATCHER = HERE / "lora_pull_watcher.py"
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
# Stubs
# --------------------------------------------------------------------------- #

SSH_STUB = r'''#!/usr/bin/env python3
"""Stub ssh: emits the probe sections described by a scenario JSON.

Process sections use procscan's wire format -- "<pid> <state>" lines then
"OK <live> <undead>" -- because that is what the box sends now. `trainers_raw`
lets a scenario emit something with no OK line, which is a probe that did not run.
"""
import json, os, sys
sc = json.load(open(os.environ["LPW_TEST_SCENARIO"]))
if sc.get("ssh_exit"):
    sys.stderr.write("stub ssh: connection refused\n")
    sys.exit(int(sc["ssh_exit"]))
epoch = sc.get("box_epoch", 1787200000)


def procsec(name, live, undead=0, raw=None):
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
print("###SENTMTIME")
if sc.get("sentinel"):
    print(sc.get("sentinel_mtime", epoch))
print("###HB"); print("step=%s loss=%s" % (sc.get("hb_step", 1850), sc.get("loss", "0.02")))
print("###HBMTIME"); print(sc.get("hb_mtime", epoch))
print("###CKPT")
for step in sc["ckpts"]:
    print("-rw-r--r-- 1 root root 131227832 1787200000 step-%d.safetensors" % step)
procsec("TRAINERS", sc.get("trainers", 3), sc.get("trainers_undead", 0),
        sc.get("trainers_raw"))
procsec("SUPERVISORS", sc.get("supervisors", 3))
print("###END")
'''

PULL_STUB = r'''#!/usr/bin/env python3
"""Stub pull_latest_lora.py: records the call, then installs the steps it is told to."""
import json, os, sys
sc = json.load(open(os.environ["LPW_TEST_SCENARIO"]))
with open(os.environ["LPW_TEST_PULL_CALLS"], "a") as fh:
    fh.write(" ".join(sys.argv[1:]) + "\n")
print("stub pull: argv=%s" % " ".join(sys.argv[1:]))
if sc.get("pull_exit"):
    print("stub pull: failing on purpose")
    sys.exit(int(sc["pull_exit"]))
dest = sys.argv[sys.argv.index("--dest-dir") + 1]
for step in sc.get("pull_installs", sc["ckpts"]):
    p = os.path.join(dest, "genpt-step-%04d.safetensors" % step)
    if not os.path.exists(p):
        open(p, "wb").write(b"x" * 32)
        print("stub pull: installed %s" % os.path.basename(p))
sys.exit(0)
'''

VASTAI_STUB = r'''#!/usr/bin/env python3
"""Stub vastai: read-only status, from the scenario."""
import json, os, sys
sc = json.load(open(os.environ["LPW_TEST_SCENARIO"]))
if sc.get("vastai_exit"):
    sys.exit(int(sc["vastai_exit"]))
print(json.dumps({"id": int(sc.get("instance", "48056192")),
                  "actual_status": sc.get("vastai_status", "running")}))
'''


def write_stub(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


class Sandbox:
    """A scratch world: artifact dir, dest dir, stubs, scenario file."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.art = root / "artifacts"
        self.dest = root / "dest"
        self.bin = root / "bin"
        for d in (self.art, self.dest, self.bin):
            d.mkdir(parents=True, exist_ok=True)
        self.ssh = write_stub(self.bin / "ssh_stub.py", SSH_STUB)
        self.pull = write_stub(self.bin / "pull_stub.py", PULL_STUB)
        self.vastai = write_stub(self.bin / "vastai_stub.py", VASTAI_STUB)
        self.scenario = root / "scenario.json"
        self.calls = root / "pull_calls.txt"
        self.psw_log = root / "psw.log"
        self.psw_lock = root / "psw.lock"

    def local(self, steps: list[int]) -> None:
        for step in steps:
            (self.dest / f"genpt-step-{step:04d}.safetensors").write_bytes(b"x" * 32)

    def run(self, scenario: dict, **env_over) -> subprocess.CompletedProcess:
        self.scenario.write_text(json.dumps(scenario))
        env = dict(os.environ)
        env.update({
            "LPW_TEST_SCENARIO": str(self.scenario),
            "LPW_TEST_PULL_CALLS": str(self.calls),
            "LPW_ARTIFACT_DIR": str(self.art),
            "LPW_LOG": str(self.art / "w.log"),
            "LPW_STATE": str(self.art / "w.state.json"),
            "LPW_LOCK": str(self.art / "w.lock"),
            "LPW_DEST_DIR": str(self.dest),
            "LPW_SSH_BIN": str(self.ssh),
            "LPW_VASTAI_BIN": str(self.vastai),
            "LPW_PULL_ARGV": json.dumps([PY, str(self.pull)]),
            "LPW_PSW_LOG": str(self.psw_log),
            "LPW_PSW_LOCK": str(self.psw_lock),
            "LPW_INTERVAL_SECS": "0",
            "LPW_RETRY_SECS": "0",
            "LPW_CONFIRM_SECS": "0",
            "LPW_MAX_CYCLES": "1",
            # These scenarios are written around step 2000 and pin their own cap,
            # so they neither depend on nor go stale with the declared one (which
            # has moved four times: 2000, 6000, 5500, 6622).
            "LPW_TARGET_STEP": "2000",
            "LPW_STATIC_SECS": "900",
        })
        env.update({k: str(v) for k, v in env_over.items()})
        return subprocess.run([PY, str(WATCHER)], capture_output=True, text=True,
                              timeout=120, env=env)

    def pull_calls(self) -> list[str]:
        try:
            return [ln for ln in self.calls.read_text().splitlines() if ln.strip()]
        except OSError:
            return []


def spawn_marker_process(marker: str) -> subprocess.Popen:
    """A live process whose cmdline contains `marker`, for pid-liveness checks."""
    return subprocess.Popen([PY, "-c", "import time; time.sleep(120)", marker],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def dead_pid() -> int:
    proc = subprocess.Popen([PY, "-c", "pass"])
    proc.wait()
    for _ in range(50):
        if not Path(f"/proc/{proc.pid}").exists():
            return proc.pid
        time.sleep(0.05)
    return proc.pid


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def scenario_noop(sb: Sandbox) -> None:
    print("\n[1] every remote checkpoint is already local -> no pull at all")
    sb.local([1800, 1825])
    r = sb.run({"ckpts": [1800, 1825], "trainers": 3})
    out = r.stdout
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("logs the no-op", "nothing to do" in out, out[-400:])
    check("does not invoke the pull", sb.pull_calls() == [], str(sb.pull_calls()))
    check("counts remote and local", "remote newest=step-1825" in out, out[-400:])


def scenario_pull(sb: Sandbox) -> None:
    print("\n[2] a new checkpoint on the box -> pulled, and reported by name")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 1825, 1850], "trainers": 3})
    out = r.stdout
    calls = sb.pull_calls()
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("invokes the pull once", len(calls) == 1, str(calls))
    check("passes --all and --dest-dir", bool(calls) and "--all" in calls[0]
          and "--dest-dir" in calls[0], str(calls))
    check("names what it will fetch", "to fetch: step-1825, step-1850" in out, out[-600:])
    check("reports what landed", "PULLED 2" in out and "genpt-step-1850" in out,
          out[-800:])
    check("installed both files",
          (sb.dest / "genpt-step-1850.safetensors").exists()
          and (sb.dest / "genpt-step-1825.safetensors").exists())
    check("logs the disk check", "disk:" in out and "GiB free" in out, out[-800:])


def scenario_disk_guard(sb: Sandbox) -> None:
    print("\n[3] not enough free space -> loud skip, and no pull")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 1825], "trainers": 3},
               LPW_DISK_BUFFER_BYTES=str(1 << 60))
    out = r.stdout
    check("exits 0 (skips the cycle, does not die)", r.returncode == 0, f"rc={r.returncode}")
    check("says NOT ENOUGH DISK", "NOT ENOUGH DISK" in out, out[-500:])
    check("raises an ALERT", "ALERT" in out, out[-500:])
    check("does not invoke the pull", sb.pull_calls() == [], str(sb.pull_calls()))
    check("leaves the file unfetched",
          not (sb.dest / "genpt-step-1825.safetensors").exists())


def scenario_pull_failure(sb: Sandbox) -> None:
    print("\n[4] the pull exits non-zero -> logged from returncode, watcher survives")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 1825], "trainers": 3, "pull_exit": 7})
    out = r.stdout
    check("watcher still exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("reads the child's real status", "pull exited 7" in out, out[-600:])
    check("says it will retry", "retrying next cycle" in out, out[-600:])


def scenario_ssh_transient(sb: Sandbox) -> None:
    print("\n[5] SSH fails while vastai says running -> retries, then gives up loudly")
    r = sb.run({"ckpts": [1800], "ssh_exit": 255, "vastai_status": "running"},
               LPW_MAX_CYCLES="0", LPW_SSH_FAILS_BEFORE_GIVEUP="3",
               LPW_SSH_FAILS_BEFORE_VASTAI="2")
    out = r.stdout
    check("tolerates the first failures", "consecutive SSH failures: 1" in out
          and "consecutive SSH failures: 2" in out, out[-800:])
    check("consults vastai read-only", "vastai says instance" in out, out[-800:])
    check("does not exit on a transient failure", "consecutive SSH failures: 3" in out,
          out[-800:])
    check("gives up with a FINAL line", "FINAL: giving up after 3" in out, out[-800:])
    check("exits non-zero on giving up", r.returncode == 1, f"rc={r.returncode}")
    check("never pulled", sb.pull_calls() == [], str(sb.pull_calls()))


def scenario_instance_stopped(sb: Sandbox) -> None:
    print("\n[6] SSH dead and vastai says stopped -> clean exit, not an endless retry")
    r = sb.run({"ckpts": [1800], "ssh_exit": 255, "vastai_status": "stopped"},
               LPW_MAX_CYCLES="0", LPW_SSH_FAILS_BEFORE_VASTAI="1")
    out = r.stdout
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("says the box is gone", "FINAL:" in out and "is 'stopped'" in out, out[-800:])
    check("names where the files are", "dest" in out, out[-800:])


def scenario_stale_evidence_does_not_end_a_live_run(sb: Sandbox) -> None:
    print("\n[7] THE REGRESSION: last run's step-2000 + last run's sentinel, live "
          "trainer -> keep pulling")
    # Exactly today's box: the run resumed from step-2000 towards a higher cap, so
    # step-2000 and a sentinel recording it are both still lying around. The first
    # version of this file exited here and pulled nothing for the whole run.
    sb.local([2000])
    r = sb.run({"ckpts": [2000, 2050, 2100], "sentinel": True, "sentinel_step": 2000,
                "sentinel_mtime": 1787200000, "hb_mtime": 1787209000,
                "box_epoch": 1787209000, "hb_step": 2106, "trainers": 3},
               LPW_TARGET_STEP="6622")
    out = r.stdout
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("does NOT declare the run over", "training is over" not in out, out[-900:])
    check("marks the old sentinel stale", "sentinel=stale" in out, out[-900:])
    check("says why", "below the cap 6622" in out, out[-900:])
    check("pulls the new cumulative checkpoints", len(sb.pull_calls()) == 1,
          str(sb.pull_calls()))
    check("names them by their cumulative numbers",
          "to fetch: step-2050, step-2100" in out, out[-900:])
    check("installed them", (sb.dest / "genpt-step-2050.safetensors").exists()
          and (sb.dest / "genpt-step-2100.safetensors").exists())
    check("reports the cap it is working towards", "step-6622" in out, out[-900:])


def scenario_a_fresh_sentinel_alone_is_not_the_end(sb: Sandbox) -> None:
    print("\n[8] sentinel at the cap but the trainer is still alive -> keep pulling")
    sb.local([1975])
    r = sb.run({"ckpts": [1975, 2000], "sentinel": True, "sentinel_step": 2000,
                "trainers": 3})
    out = r.stdout
    check("does not end on a file alone", "training is over" not in out, out[-900:])
    check("still pulls", len(sb.pull_calls()) == 1, str(sb.pull_calls()))

    # The box-side cap watcher writes its sentinel BEFORE it stops the trainer, so
    # a sentinel naming the current cap exists for a while with the run still up.
    # Freshness must stay corroboration only; the live evidence is the gate.
    print("     ...even a sentinel naming the CURRENT cap, written seconds ago")
    r = sb.run({"ckpts": [6600, 6622], "sentinel": True, "sentinel_step": 6622,
                "sentinel_mtime": 1787209000, "hb_mtime": 1787209010,
                "box_epoch": 1787209020, "hb_step": 6624, "trainers": 2},
               LPW_TARGET_STEP="6622")
    out = r.stdout
    check("sentinel reads as FRESH", "sentinel=FRESH" in out, out[-900:])
    check("and still does not end the run", "training is over" not in out, out[-900:])
    check("because a live trainer outranks any file",
          "trainers=2" in out, out[-900:])

    print("     ...and with the trainer gone but the heartbeat only 60s old")
    r = sb.run({"ckpts": [6600, 6622], "sentinel": True, "sentinel_step": 6622,
                "sentinel_mtime": 1787209000, "hb_mtime": 1787209000,
                "box_epoch": 1787209060, "trainers": 0},
               LPW_TARGET_STEP="6622", LPW_MAX_CYCLES="2")
    check("a fresh sentinel does not shortcut the static-heartbeat wait",
          "training is over" not in r.stdout, r.stdout[-900:])


def scenario_trainer_gone_confirm(sb: Sandbox) -> None:
    print("\n[9] trainer gone once, heartbeat fresh -> not the end (restart window)")
    sb.local([1800])
    r = sb.run({"ckpts": [1800], "trainers": 0, "hb_mtime": 1787200000,
                "box_epoch": 1787200060}, LPW_MAX_CYCLES="2")
    out = r.stdout
    check("counts the confirmation", "1/2 confirmations" in out, out[-900:])
    check("a 60s-old heartbeat is not 'over'", "training is over" not in out,
          out[-900:])
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")

    print("     ...and gone twice WITH a long-static heartbeat -> that is the end")
    r = sb.run({"ckpts": [1800, 2000], "trainers": 0, "hb_mtime": 1787200000,
                "box_epoch": 1787209000, "pull_installs": [1800, 2000]},
               LPW_MAX_CYCLES="0")
    out = r.stdout
    check("declares the run over", "training is over" in out, out[-1200:])
    check("quotes both halves of the evidence",
          "confirmed 2x" in out and "static for 9000s" in out, out[-1200:])
    check("does the final backfill itself", len(sb.pull_calls()) == 1,
          str(sb.pull_calls()))
    check("fetched the last checkpoint",
          (sb.dest / "genpt-step-2000.safetensors").exists())


def scenario_final_pull_is_not_delegated(sb: Sandbox) -> None:
    print("\n[10] the end, with post_stop_watcher alive -> still pulls here, no "
          "handoff promise")
    psw = spawn_marker_process("post_stop_watcher")
    try:
        sb.psw_lock.write_text(f"{psw.pid}\n")
        sb.local([1800])
        r = sb.run({"ckpts": [1800, 2000], "trainers": 0, "hb_mtime": 1787200000,
                    "box_epoch": 1787209000, "pull_installs": [1800, 2000]},
                   LPW_MAX_CYCLES="0")
        out = r.stdout
        check("exits 0", r.returncode == 0, f"rc={r.returncode}")
        check("never claims another process will finish the job",
              "handing off" not in out, out[-1200:])
        check("pulls the remainder itself", len(sb.pull_calls()) == 1,
              str(sb.pull_calls()))
        check("fetched the cap checkpoint",
              (sb.dest / "genpt-step-2000.safetensors").exists())
        check("says the instance is still billing",
              "still running and billing" in out, out[-1200:])
        check("mentions the other watcher without relying on it",
              f"pid {psw.pid}" in out and "do not assume it will act" in out,
              out[-1200:])
    finally:
        psw.kill()


def scenario_final_pull_reports_what_it_could_not_get(sb: Sandbox) -> None:
    print("\n[11] the end, but the last pull fails -> names the remote-only files")
    r = sb.run({"ckpts": [1800, 2000], "trainers": 0, "hb_mtime": 1787200000,
                "box_epoch": 1787209000, "pull_exit": 4}, LPW_MAX_CYCLES="0")
    out = r.stdout
    check("exits 0", r.returncode == 0, f"rc={r.returncode}")
    check("alerts", "could NOT be fetched" in out, out[-1400:])
    check("names both by cumulative step",
          "step-1800" in out and "step-2000" in out, out[-1400:])
    check("tells the human how to get them by hand",
          "pull_latest_lora.py --all" in out, out[-1400:])


def scenario_zombie_trainer_is_not_alive(sb: Sandbox) -> None:
    print("\n[12] a zombie trainer does not keep the run 'alive'")
    r = sb.run({"ckpts": [1800, 2000], "trainers": 0, "trainers_undead": 2,
                "hb_mtime": 1787200000, "box_epoch": 1787209000,
                "pull_installs": [1800, 2000]}, LPW_MAX_CYCLES="0")
    out = r.stdout
    check("reports them separately", "zombie/dead, not counted" in out, out[-1200:])
    check("still concludes the run is over", "training is over" in out, out[-1200:])


def scenario_unknown_count_keeps_pulling(sb: Sandbox) -> None:
    print("\n[13] the box-side count did not run -> UNKNOWN, keep pulling, never 'over'")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 2000], "trainers_raw": ["sh: python3: not found"],
                "hb_mtime": 1787200000, "box_epoch": 1787209000})
    out = r.stdout
    check("logs UNKNOWN rather than 0", "trainers=UNKNOWN" in out, out[-1200:])
    check("does not declare the run over", "training is over" not in out, out[-1200:])
    check("keeps pulling", len(sb.pull_calls()) == 1, str(sb.pull_calls()))


def scenario_psw_mid_pull(sb: Sandbox) -> None:
    print("\n[14] post_stop_watcher is mid-backfill -> yield the cycle")
    sb.psw_log.write_text(
        "[t] ----- pull_latest_lora.py --all output begins 2026-08-20T12:00:00 -----\n"
        "step-1850 -> genpt-step-1850.safetensors\n")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 1850], "trainers": 3})
    out = r.stdout
    check("yields", "mid-backfill" in out, out[-600:])
    check("does not pull", sb.pull_calls() == [], str(sb.pull_calls()))

    print("     ...and resumes once that pull has finished")
    sb.psw_log.write_text(
        "[t] ----- pull_latest_lora.py --all output begins 2026-08-20T12:00:00 -----\n"
        "[t] ----- pull output ends rc=0 2026-08-20T12:05:00 -----\n")
    r = sb.run({"ckpts": [1800, 1850], "trainers": 3})
    check("pulls again afterwards", len(sb.pull_calls()) == 1, str(sb.pull_calls()))
    check("no yield the second time", "mid-backfill" not in r.stdout, r.stdout[-400:])


def scenario_lock_live(sb: Sandbox) -> None:
    print("\n[15] another live watcher holds the lock -> refuse to start")
    other = spawn_marker_process(str(WATCHER))
    try:
        (sb.art / "w.lock").write_text(f"{other.pid}\n")
        sb.local([1800])
        r = sb.run({"ckpts": [1800, 1850], "trainers": 3})
        out = r.stdout
        check("exits 0", r.returncode == 0, f"rc={r.returncode}")
        check("says who holds it", f"already running (pid {other.pid})" in out,
              out[-600:])
        check("does nothing else", sb.pull_calls() == [], str(sb.pull_calls()))
    finally:
        other.kill()


def scenario_lock_stale(sb: Sandbox) -> None:
    print("\n[16] a lock from a dead owner -> taken over, not obeyed (power-loss case)")
    stale = dead_pid()
    (sb.art / "w.lock").write_text(f"{stale}\n")
    sb.local([1800])
    r = sb.run({"ckpts": [1800, 1850], "trainers": 3})
    out = r.stdout
    check("takes the stale lock over", f"stale lock from pid {stale}" in out, out[-600:])
    check("and does the work", len(sb.pull_calls()) == 1, str(sb.pull_calls()))
    check("releases the lock on exit", not (sb.art / "w.lock").exists())


def scenario_source_guarantees() -> None:
    print("\n[17] structural guarantees about the file itself")
    src = WATCHER.read_text()
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    check("no destroy path exists in the file", "destroy" not in body.lower(),
          "the word appears outside comments")
    check("no stop subcommand is ever built",
          '"stop"' not in body and "'stop'" not in body)
    check("nothing is piped through tee",
          "| tee" not in body and '"tee"' not in body)
    check("child status comes from returncode", "proc.returncode" in body)
    check("artifacts default outside the repo",
          'Path.home() / ".lobora"' in src and str(HERE) not in src)
    offenders = [ln for ln in src.splitlines()
                 if not ln.lstrip().startswith("#")
                 and "cmdline" in ln and "wc -l" in ln]
    check("no grep-over-/proc pipeline survives in the code", not offenders,
          str(offenders))
    check("process counting is delegated to the one shared implementation",
          "procscan.remote_scan_command" in src and "procscan.parse_section" in src
          and "procscan.alive" in src)
    check("the cap is read, not hardcoded",
          'runcap.target_step("LPW_TARGET_STEP")' in src)


# --------------------------------------------------------------------------- #

SCENARIOS = [
    scenario_noop,
    scenario_pull,
    scenario_disk_guard,
    scenario_pull_failure,
    scenario_ssh_transient,
    scenario_instance_stopped,
    scenario_stale_evidence_does_not_end_a_live_run,
    scenario_a_fresh_sentinel_alone_is_not_the_end,
    scenario_trainer_gone_confirm,
    scenario_final_pull_is_not_delegated,
    scenario_final_pull_reports_what_it_could_not_get,
    scenario_zombie_trainer_is_not_alive,
    scenario_unknown_count_keeps_pulling,
    scenario_psw_mid_pull,
    scenario_lock_live,
    scenario_lock_stale,
]


def main() -> int:
    print(f"lora_pull_watcher self-test  ({WATCHER})")
    for fn in SCENARIOS:
        with tempfile.TemporaryDirectory(prefix="lpw_selftest_") as tmp:
            fn(Sandbox(Path(tmp)))
    scenario_source_guarantees()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
