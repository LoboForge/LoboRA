# RUNBOOK — MiniMax-H3 LoRA on a rented single-GPU box

Field notes from a full Ref2VA run on one 80 GB card, via DiffSynth-Studio's
`examples/minimax_h3/model_training/train.py`. Everything here cost real time to find.
Scripts referenced live in [`scripts/vast/`](scripts/vast).

**Read these five first, they are the expensive ones:**

1. §1 — a LoRA that ComfyUI "loads" with zero patches renders as the base model.
2. §4 — resume restores **weights only**, and the restarted step counter **overwrites** earlier checkpoints.
3. §5 — `save_steps` counts **micro-batches**, not optimizer steps.
4. §7 — piping the trainer through `tee` makes `$?` always `0`, silently killing crash recovery.
5. §8 — OOM arrives at a **random** step, because sample order reshuffles every attempt.

---

## Order of operations

```bash
export WORKSPACE=/workspace RUN=my_run          # every script reads these
bash scripts/vast/bootstrap_diffsynth.sh        # diffsynth from git main + bitsandbytes
python scripts/vast/patch_diffsynth_logger.py   # step-offset + heartbeat hooks
python scripts/download_weights.py --dest $WORKSPACE/models/MiniMax-H3
python scripts/vast/rebuild_metadata.py --dataset $WORKSPACE/dataset   # metadata.json
python scripts/vast/sample_baseline_ref2va.py   # base weights generate? (A/B control)
bash scripts/vast/run_two_stage_train.sh        # stage 1 cache, then stage 2 train
```

Re-run `patch_diffsynth_logger.py` after **any** diffsynth reinstall or upgrade: pip
overwrites site-packages and takes the hooks with it, silently.

Unattended, with crash recovery, once the cache exists:

```bash
tmux new-session -d -s h3_train "bash scripts/vast/supervise_stage2.sh"
bash scripts/vast/restart_stage2.sh             # stop by exact PID + relaunch
```

Locally, to get a checkpoint into ComfyUI:

```bash
scripts/pull_latest_lora.py                     # newest; --all to backfill
```

---

## 1. ComfyUI key remap is mandatory

DiffSynth saves adapter keys as:

```
blocks.0.attn.out_proj.lora_A.default.weight
```

ComfyUI's `model_lora_keys_unet` wants:

```
diffusion_model.blocks.0.attn.out_proj.lora_A.weight
```

So: **prefix `diffusion_model.`, strip `.default`.**

Without the remap ComfyUI matches nothing, applies **0 patches**, logs only
`lora key not loaded`, and renders **exactly the base model**. Nothing errors. It looks
like training failed. Measured on this run: **104/104 patches applied after remap, 0 raw.**

`scripts/pull_latest_lora.py` does the remap while copying, rewriting only the header
and copying the tensor payload byte for byte (sha256 verified end to end).

Sanity check before blaming the training: if a LoRA sample is pixel-identical to the
baseline sample, the adapter is not being applied.

## 2. Stage 2 loads only the DiT

| Stage | Task | Loads |
|---|---|---|
| 1 | `sft:data_process` | text encoder + video VAE + audio VAE |
| 2 | `sft:train` | **only the 13 DiT shards** |

Stage 2 reads latents from the cache, so the text encoder and VAEs are dead weight
there. `scripts/vast/make_model_paths.py` emits both `--model_paths` JSONs correctly;
sharded groups must be passed as a **nested list**, ordered by shard index.

Even DiT-only, 80 GB is tight: peak **77.6 GiB of 79.14 GiB**. Naming one shard in
`--fp8_models` selects the whole group and drops the frozen base from ~62 GiB (bf16) to
~31 GiB (fp8_e4m3fn) while still computing in bf16.

## 3. VRAM: what actually bought headroom

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — **necessary but not sufficient.**
  ~861 MiB of fragmentation survived it, and one OOM still missed by ~60 MiB.
- `--use_gradient_checkpointing_offload` — **the real lever.** Pushes checkpointed
  activations to host RAM. Costs ~15–30% step time. This is what made the run survive.

Order of attack: fp8 the frozen base, then `expandable_segments`, then offload.

## 4. Resume is weights-only, and will overwrite your checkpoints

`--lora_checkpoint` restores **adapter weights only**. It does *not* restore:

- the step counter — the progress bar restarts at 0 while the weights carry forward
- Adam moments — the optimizer warm-starts
- the LR schedule position

**The trap:** with the counter back at 0, the next save is `step-100` again, and it
**silently overwrites the previous attempt's `step-100`** — same name, different weights.
Three crashes and you have one attempt's worth of files.

`scripts/vast/supervise_stage2.sh` works around this by exporting
`DIFFSYNTH_STEP_OFFSET` from the resumed checkpoint's number, so saves continue the
lineage (`step-700`, `step-800`, ...). That is **numbering only** — it does not restore
optimizer state. Stock DiffSynth ignores that variable; it only takes effect once
`scripts/vast/patch_diffsynth_logger.py` has been run against the install.

The supervisor also **validates** a checkpoint before resuming from it: parse the
safetensors header, confirm `8 + header_len + payload == file size`, confirm it contains
`lora` keys. A checkpoint written when the crash landed is truncated and will not load.

## 5. Step units: micro-batches vs optimizer steps

`save_steps` and the tqdm total count **micro-batches**, not optimizer steps.

With `dataset_repeat 7`, `gradient_accumulation_steps 4`, 963 cached samples:

```
963 × 7          = 6741 micro-steps   <- what tqdm shows
6741 / 4         = 1685 optimizer steps
save_steps 100   = every 25 optimizer steps
```

This units mismatch produced a confident, wrong conclusion that the config was being
ignored. Do the division before believing the bar.

## 6. Set `save_steps` low enough to survive a crash

At `save_steps 100`, an attempt that OOM'd at step 80 contributed **nothing at all** —
hours of compute, zero artifacts. Pick a value where the worst case loses minutes, not
an attempt. This run settled on **25**.

## 7. The `tee` + `$?` trap

Piping the trainer through `tee` (to see it live in the pane *and* keep a log) makes
`$?` report **tee's** status, which is always `0`. Every crash then looks like success
and the supervisor exits happy while training is dead.

```bash
bash "$INNER" $CKPT 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}          # must be the VERY NEXT LINE
```

Use `${PIPESTATUS[0]}` and/or `set -o pipefail`, and **test it** with a stub inner
script that exits non-zero — do not assume:

```bash
printf '#!/usr/bin/env bash\nexit 7\n' > /tmp/stub.sh && chmod +x /tmp/stub.sh
H3_INNER=/tmp/stub.sh H3_MAX_ATTEMPTS=2 H3_RETRY_SLEEP=0 \
  bash scripts/vast/supervise_stage2.sh      # must report rc=7, not rc=0
```

## 8. OOM is a random-arrival hazard

The trainer builds its loader as `DataLoader(dataset, shuffle=True, ...)` with no
explicit generator (`diffsynth/diffusion/runner.py`), so **sample order is reshuffled
every attempt**. An OOM at step 80 is not
reproducible at step 80; it is one unlucky heavy sample arriving whenever it arrives.

Consequences: do not tune against a specific step number, do not conclude "it always
dies at N", and expect a retry to get further purely by luck. Fix headroom, not the step.

## 9. Two-stage caching

Stage 1 encodes the dataset to `$OUT/split-cache`. For this run: **963 files, ~8 hours,
~150 GB**. Then:

- Re-running stage 1 **only verifies** an existing cache; it does not re-encode. Safe
  after a crash, and cheap.
- Changing `height`, `width` or `num_frames` **invalidates** the cache. Budget another
  full stage 1 before changing shot geometry.
- The cache is the expensive asset on an ephemeral box. It is the thing to protect.

## 10. Operating the box without shooting the run

- **Never `pkill` / `killall`.** The pattern that matches the trainer also matches
  sibling jobs. Capture exact PIDs, re-captured fresh each time.
- **Kill the supervisor first**, or it reads the trainer's death as a crash and
  immediately relaunches into a half-dead GPU.
- `SIGTERM`, wait up to ~60s, and only then `SIGKILL` — a hard kill mid-save leaves a
  truncated checkpoint.
- **Wait for VRAM to drain** (`nvidia-smi` under ~4 GB) before relaunching, or the new
  attempt OOMs instantly on the old attempt's memory.
- Run under `tmux`; an SSH drop otherwise takes the run with it.

`scripts/vast/restart_stage2.sh` does all of the above in order.

To check a live run without trawling a multi-GB log, read the heartbeat the patched
logger writes (step, loss, VRAM peak, headroom, attempt):

```bash
cat $WORKSPACE/logs/h3_heartbeat.txt
tail -5 $WORKSPACE/logs/h3_supervisor.log
```

## 11. Pulling checkpoints off the box

`scripts/pull_latest_lora.py` is read-only on the box and:

- refuses a file below a plausible size, and re-`stat`s after a settle window, so a
  **mid-write** checkpoint is never copied
- `sha256`s on the box and locally and compares
- remaps keys for ComfyUI, then re-verifies tensor count, contiguous offsets and total
  size before installing
- installs atomically (`os.replace`) into the ComfyUI LoRA dir as `genpt-step-NNNN`

Vast **reassigns the SSH port on every instance restart**. Re-resolve with
`vastai show instances`, then `LORA_SSH_PORT=<newport> scripts/pull_latest_lora.py`.

## Privacy

Dataset captions, images and video are never read for inspection, never printed, and
never sent to any vision or multimodal model. `rebuild_metadata.py` reads sidecar
captions only to place them in `metadata.json` and reports **counts only**.
