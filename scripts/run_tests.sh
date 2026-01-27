#!/bin/bash

BRANCH_NAME=$1
# If $2 starts with / or ., assume it's a venv path and use default target dir
if [[ "$2" == /* ]] || [[ "$2" == .* ]]; then
    VENV_BASE_DIR=$2
    TARGET_DIR="terratorch.$BRANCH_NAME"
else
    TARGET_DIR=${2:-"terratorch.$BRANCH_NAME"}
    VENV_BASE_DIR=$3
fi

BASE_PATH=$(pwd)

# 1. Validation ---
if [ -z "$BRANCH_NAME" ]; then
    echo "Usage: $0 <branch_name> [venv_base_path]"
    echo "Usage: $0 <branch_name> [custom_target_dir] [venv_base_path]"
    exit 1
fi

# 2. Safety Check: Abort if already inside a Git Repo ---
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: You are currently inside a git repository."
    echo "Run this from a clean parent directory."
    exit 1
fi

# 3. Path Setup ---
# Create the checkout folder automatically
mkdir -p "$TARGET_DIR"
FULL_PATH=$(cd "$TARGET_DIR" && pwd)

if [ -n "$VENV_BASE_DIR" ]; then
    mkdir -p "$VENV_BASE_DIR"
    # Resolve absolute path for venv to ensure LSF finds it
    VENV_ROOT=$(cd "$VENV_BASE_DIR" && pwd)
    VENV_PATH="$VENV_ROOT/venv_$BRANCH_NAME"
else
    VENV_PATH="$FULL_PATH/.venv"
fi

# 4. Clone / Checkout Logic ---
echo "Cloning Branch: $BRANCH_NAME into $FULL_PATH ---"
if [ ! -d "$FULL_PATH/.git" ]; then
    git clone git@github.com:terrastackai/terratorch.git "$FULL_PATH"
fi

cd "$FULL_PATH" || exit
git fetch origin
git checkout "$BRANCH_NAME" || git checkout -b "$BRANCH_NAME" "origin/$BRANCH_NAME"

# 5. Virtual Environment Setup ---
echo "Setting up Virtual Environment at: $VENV_PATH ---"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Installing Dependencies (this may take a minute) ---"
pip install --upgrade pip
if ! pip install -e ".[test]"; then
    echo "Error: pip install failed."
    exit 1
fi

# 6. LSF Submission ---
echo "Submitting to LSF (bsub) ---"
bsub -gpu "num=1" -Is -R "rusage[ngpus=1, cpu=4, mem=128GB]" \
     -J "terratorch_ci_$BRANCH_NAME" \
     "/bin/bash -c 'source $VENV_PATH/bin/activate && cd $FULL_PATH && pytest ./tests'"
