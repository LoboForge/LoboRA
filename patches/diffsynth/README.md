# DiffSynth-Studio source edits the H3 run depends on

Rescued off a live Vast training box so they survive the instance being destroyed. These
edits existed in **no git remote and no local copy** — the working H3 LoRA training run
depends on them.

**There are two patch files, and they go to two different places.** That is the whole
point of this directory, and getting it wrong produces a box that trains for hours and
then blows up on VRAM with no error message. Read [Which tree gets
what](#which-tree-gets-what) before applying anything.

| Patch | Target tree | Contains |
|---|---|---|
| [`checkout/examples_minimax_h3_train.diff`](checkout/examples_minimax_h3_train.diff) | the **git checkout** of DiffSynth-Studio | `examples/minimax_h3/model_training/train.py` |
| [`site-packages/diffsynth_diffusion.diff`](site-packages/diffsynth_diffusion.diff) | the **venv `site-packages`** install of the `diffsynth` package | `diffsynth/diffusion/{logger,runner,training_module}.py` |

## Which tree gets what

`scripts/vast/bootstrap_diffsynth.sh` creates **two independent copies** of DiffSynth:

1. a **non-editable** `pip install "git+https://github.com/modelscope/DiffSynth-Studio.git@<sha>"`,
   which lands a snapshot of the `diffsynth` package in the venv's `site-packages`;
2. a separate `git clone` into `$WORKSPACE/DiffSynth-Studio`, because the MiniMax-H3
   training *example* ships in the repo and not in the wheel.

Because the install is non-editable, those two trees share no files. Editing one does not
touch the other. Which means:

- `accelerate launch examples/minimax_h3/model_training/train.py` executes the **checkout's**
  copy of `train.py` — so the `train.py` fix belongs to the checkout.
- that script then does `from diffsynth.diffusion import *`, which resolves through
  `sys.path` to **`site-packages`** — so all three `diffsynth/diffusion/*` fixes belong to
  site-packages, and applying them to the checkout accomplishes nothing at all.

This was verified on the live box rather than reasoned about: the running trainer writes
heartbeat lines, and `_write_heartbeat` exists **only** in the site-packages copy of
`diffsynth/diffusion/logger.py`. The process was demonstrably importing site-packages.

Resolve the site-packages path at runtime instead of hardcoding it — it moves with the
venv, the Python version and the platform:

```bash
SITE=$("$PYTHON" -c 'import diffsynth, os; print(os.path.dirname(os.path.dirname(diffsynth.__file__)))')
```

### Apply

```bash
LOBORA=/workspace/LoboRA        # this repo
DIFFSYNTH=/workspace/DiffSynth-Studio
PYTHON=/workspace/venv/bin/python

# 1. the executed example script -> the CHECKOUT
git -C "$DIFFSYNTH" apply --check -v "$LOBORA/patches/diffsynth/checkout/examples_minimax_h3_train.diff"
git -C "$DIFFSYNTH" apply       -v "$LOBORA/patches/diffsynth/checkout/examples_minimax_h3_train.diff"

# 2. the IMPORTED package -> SITE-PACKAGES (not a git tree, so use `patch`)
SITE=$("$PYTHON" -c 'import diffsynth, os; print(os.path.dirname(os.path.dirname(diffsynth.__file__)))')
patch -p1 --dry-run -d "$SITE" < "$LOBORA/patches/diffsynth/site-packages/diffsynth_diffusion.diff"
patch -p1           -d "$SITE" < "$LOBORA/patches/diffsynth/site-packages/diffsynth_diffusion.diff"

# 3. ops hooks (step offset + heartbeat), also into the imported package
"$PYTHON" "$LOBORA/scripts/vast/patch_diffsynth_logger.py"

# 4. prove the fp8 fix reached the module that actually gets imported. Without this,
#    a missing fp8 fix is invisible until the GPU runs out of memory hours later.
"$PYTHON" "$LOBORA/scripts/vast/verify_diffsynth_patches.py"
```

`site-packages` is not a git work tree, so `git apply` there needs `--unsafe-paths` and
still will not stage anything useful; plain `patch -p1 -d` is the right tool. Both patches
are `-p1` with `a/`…`b/` prefixes, and both are idempotency-unsafe — `patch` will ask
interactively if you apply twice, so always dry-run first.

## Upstream base

| | |
|---|---|
| Remote | `https://github.com/modelscope/DiffSynth-Studio.git` |
| Base commit | `03615819a6209a198c7e4020988a18ba64e05fb0` (`0361581`) |
| Commit subject | `Support FLUX.1 Fill Redux InsertAnything (#1611)` |

`bootstrap_diffsynth.sh` pins **both** the pip install and the clone to that sha, so the
two trees stay in lockstep and these patches keep applying. Both files were verified with
`git apply --check` against a pristine upstream checkout at that exact commit: they are
reapplicable, not archival.

If you deliberately want a newer upstream, expect fuzz: use `git apply -3` / `patch -p1
--merge`, then re-read the per-file notes below and confirm each fix is still necessary —
some may have been fixed upstream.

## Per-file explanation

### 1. `examples/minimax_h3/model_training/train.py` — **CHECKOUT.** Load-bearing.

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

Fails loudly, which is why this one has never been mis-targeted: put it in the wrong tree
and the run refuses to start.

### 2. `diffsynth/diffusion/training_module.py` — **SITE-PACKAGES.** fp8 / offload on sharded models.

**The dangerous one.** It is the only edit here whose absence produces no error at all.

In `parse_model_configs`, `model_paths` entries may be either a single path string *or* a
list of shard paths for one sharded model. Upstream tests membership with
`path in fp8_models`, which can never match for a sharded entry, because `path` is then a
*list* while `--fp8_models` / `--offload_models` name individual shard files. Result: fp8
quantisation and CPU offload silently do not apply to sharded models. No warning, no
traceback — the frozen 13-shard DiT just loads in bf16 at ~62 GiB instead of fp8_e4m3fn at
~31 GiB, and the run dies of an "inexplicable" OOM on an 80 GiB card.

The fix normalises to `shards = path if isinstance(path, list) else [path]` and matches with
`any(...)`, so naming any one shard selects the whole group.

`scripts/vast/verify_diffsynth_patches.py` exists specifically for this failure mode: it
asserts the fixed expression is present in the source of the **imported**
`diffsynth.diffusion.training_module` and exits non-zero if it is not. Run it after every
install, upgrade or reinstall.

### 3. `diffsynth/diffusion/runner.py` — **SITE-PACKAGES.** Data-preprocessing resumability.

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

### 4. `diffsynth/diffusion/logger.py` — **SITE-PACKAGES.** Superseded on the new code path.

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

Note that this file is *also* the one `scripts/vast/patch_diffsynth_logger.py` edits, and
the two overlap: the script's step-offset hunk is the same change as this one. The script is
idempotent and keys off `_write_heartbeat` — its own unique contribution — precisely so that
a tree already carrying this patch does not read as "already patched" and lose the
heartbeat. Applying the patch and then running the script gives the intended result in
either order.

### 5. Optional: `diffsynth/core/data/unified_dataset.py` — **SITE-PACKAGES.** OOM escape hatch.

[`site-packages/diffsynth_cache_exclude.diff`](site-packages/diffsynth_cache_exclude.diff)
is **not** part of the rescued set — it was authored in this repo — and is only needed if
stage 2 OOMs on a small number of unusually long samples. See
[RUNBOOK §13](../../RUNBOOK.md) and
[`scripts/vast/regen_cache_exclusions.py`](../../scripts/vast/regen_cache_exclusions.py).

## Summary

| File | Tree | Failure mode if missing |
|---|---|---|
| `examples/minimax_h3/model_training/train.py` | checkout | Run won't start: `ValueError: No valid model files` |
| `diffsynth/diffusion/training_module.py` | site-packages | **Silent.** fp8/offload ignored for sharded models → VRAM blowup with no error |
| `diffsynth/diffusion/runner.py` | site-packages | Preprocessing restarts from scratch; one bad sample kills the pass |
| `diffsynth/diffusion/logger.py` | site-packages | Checkpoints restart at `step-100` and overwrite |
| `diffsynth/core/data/unified_dataset.py` | site-packages (optional) | No way to exclude an over-long sample without rebuilding the cache |

## Provenance / hygiene

- Box access was **strictly read-only**: `git status`, `git diff`, `git log`, `git rev-parse`,
  `git remote -v`, `sed -n`, `grep`, `ls`, `md5sum`. No writes, no edits, no signals, no
  `checkout`/`stash`/`clean`. Live training untouched.
- Secret scan over both diffs (api keys, tokens, bearer, `hf_*`, `sk-*`, `AKIA*`, `ghp_*`,
  PEM headers): **nothing found.** They are Python source only.
- No dataset media, captions, or filenames were read or included. No vision/multimodal
  model was used at any point.
- **Deliberately excluded** as untracked noise, not run dependencies:
  `diffsynth/diffusion/logger.py.bak_pre_stepoffset`,
  `diffsynth/diffusion/training_module.py.bak_pre_fp8`. These are pre-edit backups; the
  patch supplies their post-edit content.

On-box md5sums of the modified files at capture time:

```
b967de71572cc2ccd7a36fa66c893bb8  diffsynth/diffusion/logger.py
d5963e067c42ba626981bac66796606f  diffsynth/diffusion/runner.py
fdc59385cc2fd9ee81b86aad201854cd  diffsynth/diffusion/training_module.py
354d3dfe7371056e70d27c552fc812f0  examples/minimax_h3/model_training/train.py
```
