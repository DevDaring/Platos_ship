#!/usr/bin/env bash
# vm_bootstrap.sh — install the exact tested environment. Runs ON the VM.
#
# Ubuntu's Python is not the one the tests passed on, so uv installs
# Python 3.13 and the pinned versions in requirements.lock.txt. The suite then
# runs on the VM itself: a green suite here means the VM computes what the
# local machine computed.

set -euo pipefail

ROOT="$HOME/platos/Code/Code_Phase_3"
VENV="$HOME/platos/.venv"
cd "$ROOT"

say() { printf '\n==> %s\n' "$*"; }

say "System packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git rsync curl ca-certificates >/dev/null

say "uv + Python 3.13"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13 >/dev/null
[ -d "$VENV" ] || uv venv --python 3.13 "$VENV" >/dev/null
# shellcheck disable=SC1091
source "$VENV/bin/activate"

say "Pinned packages"
uv pip install -q -r requirements.lock.txt
python -c "import pandas, pyarrow, numpy, scipy, openai; \
print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__, \
'| numpy', numpy.__version__, '| openai', openai.__version__)"

say "Secrets file"
chmod 600 "$HOME/platos/Code/.env"
ls -l "$HOME/platos/Code/.env"

say "Test suite"
python -m pytest -q 2>&1 | tail -3

say "Plan"
python run_all.py --list 2>&1 | tail -6
