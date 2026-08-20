# DiffSynth-Studio hand-modifications from the Vast training box

Rescued off the live Vast box (`ssh9.vast.ai:16192`, `/workspace/DiffSynth-Studio`) so they
survive the instance being destroyed. These four edits existed in **no git remote and no
local copy** — the working H3 LoRA training run depends on them.

Companion patch: [`diffsynth_box_hand_patches.diff`](diffsynth_box_hand_patches.diff) (same directory).

## Upstream base

| | |
|---|---|
| Remote | `https://github.com/modelscope/DiffSynth-Studio.git` (origin, fetch+push) |
| Branch | `main` |
| Base commit | `03615819a6209a198c7e4020988a18ba64e05fb0` (`0361581`) |
| Commit subject | `Support FLUX.1 Fill Redux InsertAnything (#1611)` |
| Clone type | shallow (`.git/shallow` pins that same sha, depth 1) |

The patch was verified with `git apply --check` against a **pristine** upstream checkout at
that exact commit — all four files apply cleanly. It is reapplicable, not just archival.

## Reapply on a fresh box

```bash
cd /workspace
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
# Pin the exact commit the patch was authored against.
git checkout 03615819a6209a198c7e4020988a18ba64e05fb0

PATCH=/workspace/LoboRA/patches/diffsynth/diffsynth_box_hand_patches.diff
git apply --check -v "$PATCH"   # dry run first
git apply -v "$PATCH"
```

If you only need a shallow tree (faster, matches how the box was set up):

```bash
mkdir -p /workspace/DiffSynth-Studio && cd /workspace/DiffSynth-Studio
git init .
git remote add origin https://github.com/modelscope/DiffSynth-Studio.git
git fetch --depth 1 origin 03615819a6209a198c7e4020988a18ba64e05fb0
git checkout FETCH_HEAD
git apply -v /workspace/LoboRA/patches/diffsynth/diffsynth_box_hand_patches.diff
```

If you deliberately want a newer upstream `main` instead, expect fuzz: use
`git apply -3` (3-way merge) or `patch -p1 --merge`, then re-read the per-file notes below
and confirm each fix is still necessary — some may have been fixed upstream.

## Per-file explanation

### 1. `examples/minimax_h3/model_training/train.py` — **STILL NEEDED. Load-bearing.**

The most important one, and the hardest to rediscover. Without it the run does not start
at all.

`parse_path_or_model_id()` returns `ModelConfig(path=...)` when the argument is an existing
local directory, and `ModelConfig(model_id=..., origin_file_pattern=...)` only for the
`"repo/id:pattern"` form. Upstream then *reconstructs* the config as
`ModelConfig(model_id=processor_config.model_id, origin_file_pattern=...)`, **discarding
`path=`**. For a local processor directory (e.g. `Ref2VA/processor`) both `model_id` and
`origin_file_pattern` are `None`, so `ModelConfig.check_input()` raises:

> `ValueError: No valid model files. Please use ModelConfig(path="xxx") or ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz").`

The fix passes the already-parsed config straight through (preserving `path=`), sets
`skip_download = True` so it does not try to reach the Hub/ModelScope for a model that is
already on disk, and adds an explicit guard that raises a *comprehensible* error if the
processor path resolved to neither a path nor a model_id. `skip_download` is a real field on
`ModelConfig` and, per upstream's own error text, is only supported alongside `path=` — so
the fix is consistent with upstream intent. Verified against
`diffsynth/core/loader/config.py:44-65`.

### 2. `diffsynth/diffusion/runner.py` — **STILL NEEDED.** Data-preprocessing resumability.

Rewrites `launch_data_process_task` from a DataLoader-driven loop into an index-driven loop
over `range(len(dataset))`. Two independent behaviour changes:

- **Skip-if-cached.** The output path `{output}/{process_index}/{data_id}.pth` is checked
  *before* the sample is loaded, so an interrupted preprocessing pass resumes instead of
  redoing everything. Under the old DataLoader loop the sample was materialised (media
  decode + VAE encode) before anything looked at the destination.
- **Per-sample fault tolerance.** Each sample is wrapped in `try/except`; a failure prints
  `[data_process] SKIP data_id=... due to ...` and continues rather than aborting the whole
  pass. Ends with a `skipped_existing=/processed=/failed=/total=` summary line.

Incidental: `accelerator.prepare()` is no longer applied to the dataloader (there isn't one
any more), only to the model; and the `accelerator.accumulate(model)` wrapper is dropped,
which is harmless here since this is a `no_grad` caching pass with no optimizer step.
`os` was already imported, so the patch is self-contained.

### 3. `diffsynth/diffusion/training_module.py` — **STILL NEEDED.** fp8 / offload on sharded models.

In `parse_model_configs`, `model_paths` entries may be either a single path string *or* a
list of shard paths for one sharded model. Upstream tests membership with
`path in fp8_models`, which can never match for a sharded entry, because `path` is then a
*list* while `--fp8_models` / `--offload_models` name individual shard files. Result: fp8
quantisation and CPU offload silently do not apply to sharded models — you would not get an
error, just unexpectedly high VRAM use.

The fix normalises to `shards = path if isinstance(path, list) else [path]` and matches with
`any(...)`, so naming any one shard selects the whole group. The neighbouring
`training_module.py.bak_pre_fp8` backup on the box confirms this was introduced while
getting fp8 working.

### 4. `diffsynth/diffusion/logger.py` — **SUPERSEDED** (on the new code path only).

Seeds `ModelLogger.num_steps` from `int(os.environ.get("DIFFSYNTH_STEP_OFFSET") or 0)`
instead of `0`, so a warm-started run continues the checkpoint lineage
(`step-700`, `step-800`, …) rather than restarting at `step-100` and clobbering the previous
run's files. (`os` was already imported.)

**Superseded by** [`lobora/diffsynth_resume.py`](../../lobora/diffsynth_resume.py) +
[`scripts/train_h3_resumable.py`](../../scripts/train_h3_resumable.py), which derive the
cumulative step from resume state on disk. That code
*explicitly ignores* the env var — `diffsynth_resume.py` logs
`"ignoring DIFFSYNTH_STEP_OFFSET=...; the resume state on disk says ..."`, and
`scripts/supervise_h3_resumable.sh` does `unset DIFFSYNTH_STEP_OFFSET`.

**Caveat — do not delete it blindly.** The *older* supervisor
`scripts/vast/supervise_stage2.sh` still does `export DIFFSYNTH_STEP_OFFSET="$STEP_OFFSET"`.
The hand-edit is obsolete only if you launch via the resumable path. If you fall back to
`supervise_stage2.sh`, this edit is still required or checkpoints will overwrite each other.
It is harmless to apply either way: with the env var unset it evaluates to `0`, i.e.
identical to upstream. Recommendation: **keep it in the patch**, retire it when
`supervise_stage2.sh` is retired.

## Summary

| File | Status | Failure mode if missing |
|---|---|---|
| `examples/minimax_h3/model_training/train.py` | **Still needed (critical)** | Run won't start: `ValueError: No valid model files` |
| `diffsynth/diffusion/runner.py` | **Still needed** | Preprocessing restarts from scratch; one bad sample kills the pass |
| `diffsynth/diffusion/training_module.py` | **Still needed** | fp8/offload silently ignored for sharded models → VRAM blowup |
| `diffsynth/diffusion/logger.py` | **Superseded** on resumable path; still needed by `supervise_stage2.sh` | Checkpoints restart at `step-100` and overwrite |

## Provenance / hygiene

- Box access was **strictly read-only**: `git status`, `git diff`, `git log`, `git rev-parse`,
  `git remote -v`, `sed -n`, `grep`, `ls`, `md5sum`. No writes, no edits, no signals, no
  `checkout`/`stash`/`clean`. Live training untouched.
- Secret scan over the diff (api keys, tokens, bearer, `hf_*`, `sk-*`, `AKIA*`, `ghp_*`,
  PEM headers): **nothing found.** The diff is Python source only.
- No dataset media, captions, or filenames were read or included. No vision/multimodal
  model was used at any point.
- **Deliberately excluded** as untracked noise, not run dependencies:
  `diffsynth/diffusion/logger.py.bak_pre_stepoffset`,
  `diffsynth/diffusion/training_module.py.bak_pre_fp8`. These are pre-edit backups; the
  patch supplies their post-edit content. There were **no other untracked files**
  (`git status --porcelain --untracked-files=all` returned only the four modified files and
  those two `.bak`s), so nothing else needed rescuing.

On-box md5sums of the modified files at capture time:

```
b967de71572cc2ccd7a36fa66c893bb8  diffsynth/diffusion/logger.py
d5963e067c42ba626981bac66796606f  diffsynth/diffusion/runner.py
fdc59385cc2fd9ee81b86aad201854cd  diffsynth/diffusion/training_module.py
354d3dfe7371056e70d27c552fc812f0  examples/minimax_h3/model_training/train.py
```
