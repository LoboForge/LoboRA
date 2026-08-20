#!/usr/bin/env bash
# Install DiffSynth-Studio (from git, not PyPI -- the MiniMax-H3 training example is
# not in a release) plus bitsandbytes, into the venv this box will train from.
#
# This leaves TWO independent DiffSynth trees on the box, and that is deliberate:
#
#   site-packages   the non-editable pip install. This is what `import diffsynth`
#                   resolves to, so it is what the trainer actually runs.
#   $DIFFSYNTH      a git clone, because the training EXAMPLE ships in the repo and
#                   not in the wheel. Only `examples/.../train.py` is used from here.
#
# Both are pinned to the same sha so patches/diffsynth/ keeps applying to both. Floating
# on `main` was how the checkout and the install silently drifted apart.
#
# Deliberately NOT `|| true`: a half-installed diffsynth fails hours later inside the
# trainer instead of here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

# The commit patches/diffsynth/ was authored and `git apply --check`ed against.
DIFFSYNTH_REF=${DIFFSYNTH_REF:-03615819a6209a198c7e4020988a18ba64e05fb0}
DIFFSYNTH_REPO=${DIFFSYNTH_REPO:-https://github.com/modelscope/DiffSynth-Studio.git}

"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install "git+${DIFFSYNTH_REPO}@${DIFFSYNTH_REF}" bitsandbytes

if [ ! -d "$DIFFSYNTH" ]; then
  git clone "$DIFFSYNTH_REPO" "$DIFFSYNTH"
fi
git -C "$DIFFSYNTH" fetch --depth 1 origin "$DIFFSYNTH_REF"
git -C "$DIFFSYNTH" checkout -q FETCH_HEAD

"$PYTHON" - <<'PY'
import diffsynth, os
print("diffsynth", diffsynth.__file__)
print("site-packages", os.path.dirname(os.path.dirname(diffsynth.__file__)))
PY

cat <<EOF

Next, apply the source edits -- TWO patches, TWO trees. See patches/diffsynth/README.md.
  git -C $DIFFSYNTH apply <repo>/patches/diffsynth/checkout/examples_minimax_h3_train.diff
  SITE=\$($PYTHON -c 'import diffsynth,os;print(os.path.dirname(os.path.dirname(diffsynth.__file__)))')
  patch -p1 -d "\$SITE" < <repo>/patches/diffsynth/site-packages/diffsynth_diffusion.diff
  $PYTHON $HERE/patch_diffsynth_logger.py
  $PYTHON $HERE/verify_diffsynth_patches.py    # fails LOUDLY if the fp8 fix is missing
EOF
echo DIFFSYNTH_DONE
