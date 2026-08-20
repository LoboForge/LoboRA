#!/usr/bin/env bash
# Install DiffSynth-Studio (git main, not PyPI -- the MiniMax-H3 training example is
# not in a release) plus bitsandbytes, into the venv this box will train from.
#
# Deliberately NOT `|| true`: a half-installed diffsynth fails hours later inside the
# trainer instead of here.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/vast/h3_env.sh
source "$HERE/h3_env.sh"

"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install "git+https://github.com/modelscope/DiffSynth-Studio.git" bitsandbytes

# The training example lives in the checkout, not in the wheel, so it is needed too.
if [ ! -d "$DIFFSYNTH" ]; then
  git clone https://github.com/modelscope/DiffSynth-Studio.git "$DIFFSYNTH"
fi

"$PYTHON" - <<'PY'
import diffsynth
print("diffsynth", diffsynth.__file__)
PY
echo DIFFSYNTH_DONE
