#!/usr/bin/env bash
# Rebuild everything that is kept out of git so the project can run again:
#   - the Python virtual environment (.venv)
#   - the official OpenAI GPT-2 weights (models/gpt2/124M and models/gpt2/355M)
#
# Every download is checked against its SHA-256, so you get exactly the files
# this project was developed with. The same files can also be fetched from
# Python with gpt2_from_scratch.downloader.download_and_load_gpt2().
#
# Usage:
#   ./setup.sh               venv + GPT-2 124M and 355M weights (~1.9 GB)
#   ./setup.sh --small       venv + only the 124M weights (~0.5 GB)
#   ./setup.sh --no-weights  venv only
#   ./setup.sh --test        same as default, then run the smoke tests
#
# Not downloadable (trained locally, never published):
#   models/pretrained/*.pth and models/finetuning/*.pth -> keep them, or
#   retrain them with the notebooks if they are missing.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="3.10"
BASE_URL="https://openaipublic.blob.core.windows.net/gpt-2/models"

# size/file | sha256
WEIGHTS_124M=(
  "124M/checkpoint|dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb"
  "124M/encoder.json|196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
  "124M/hparams.json|d9d56e4121c427164e0c55c6f03c08e1daf9002b9b672825112d19097b680318"
  "124M/model.ckpt.data-00000-of-00001|2060c885360cc0cf41d7a6dbc4d24b5127aae20260c8b5ae521b5a6578407118"
  "124M/model.ckpt.index|71916f763f9746f9b2a06b12d91996cf1084ae008d0424543d39391c5f2dc687"
  "124M/model.ckpt.meta|4668c448fa11531fd6700460487f73e82d3272960cea942252f8744bf225c77b"
  "124M/vocab.bpe|1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
)
WEIGHTS_355M=(
  "355M/checkpoint|dd1b025d2e155283f5e300ce95bf6d5b6bc0f7fe010db73daa6975eb896ab9cb"
  "355M/encoder.json|196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
  "355M/hparams.json|310dfdb1f0c786b79d93459eaa470f3978d873bdc3e1f9fed1cad24d9dd0b1f6"
  "355M/model.ckpt.data-00000-of-00001|22b4212602a087d1cdeec3509648f4d7b1f181eaa33ce0f3c688fbfb732d622f"
  "355M/model.ckpt.index|4490e7b1f5ec0abfc1e111fcd27cdda7cb280c597ba0a885c364134e62ce43aa"
  "355M/model.ckpt.meta|06e1a92057d2a4fbdc483bd1f61a26cc274f1c999b84c6d9e9c50aa8d5ead44a"
  "355M/vocab.bpe|1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
)

WEIGHTS="all"
RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --small)      WEIGHTS="small" ;;
    --no-weights) WEIGHTS="none" ;;
    --test)       RUN_TESTS=1 ;;
    -h|--help)    sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see ./setup.sh --help)"; exit 1 ;;
  esac
done

sha256() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}

fetch() {
  local entry="$1" remote expected path tmp
  IFS='|' read -r remote expected <<< "$entry"
  path="models/gpt2/$remote"

  if [[ -f "$path" ]]; then
    if [[ "$(sha256 "$path")" == "$expected" ]]; then echo "  ok $path"; return 0; fi
    echo "  checksum mismatch for $path, downloading again"
    rm -f "$path"
  fi

  mkdir -p "$(dirname "$path")"
  tmp="$path.part"
  echo "  downloading $remote ..."
  curl -fL --retry 3 -C - -o "$tmp" "$BASE_URL/$remote"
  if [[ "$(sha256 "$tmp")" != "$expected" ]]; then
    rm -f "$tmp"
    echo "ERROR: checksum mismatch for $remote"
    return 1
  fi
  mv "$tmp" "$path"
  echo "  ok $path (verified)"
}

echo "==> Python environment"
PY="$(command -v "python$PYTHON_VERSION" || true)"
if [[ -z "$PY" ]]; then
  echo "ERROR: python$PYTHON_VERSION not found. Install it with: brew install python@$PYTHON_VERSION"
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

echo "==> OpenAI GPT-2 weights"
case "$WEIGHTS" in
  all)   for f in "${WEIGHTS_124M[@]}" "${WEIGHTS_355M[@]}"; do fetch "$f"; done ;;
  small) for f in "${WEIGHTS_124M[@]}"; do fetch "$f"; done ;;
  none)  echo "  skipped (--no-weights)" ;;
esac

echo "==> Locally trained weights"
for f in models/pretrained/model.pth models/pretrained/model_and_optimizer.pth models/pretrained/pretraining124M.pth \
         models/finetuning/spam_classifier.pth models/finetuning/spam_classifier_gpt124M.pth \
         models/finetuning/gpt2-medium355M-sft.pth models/finetuning/instruction_finetuning_gpt2-medium355M-sft.pth \
         models/finetuning/instruction355M.pth; do
  if [[ -f "$f" ]]; then echo "  ok $f"; else echo "  missing $f (not downloadable, retrain with the notebooks)"; fi
done

if [[ "$RUN_TESTS" == 1 ]]; then
  echo "==> Smoke tests"
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
fi

echo
echo "Setup complete. Activate the environment with: source .venv/bin/activate"
