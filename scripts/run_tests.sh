#!/bin/bash

BRANCH_NAME=$1
TARGET_DIR=${2:-"terratorch.$BRANCH_NAME"}
VENV_BASE_DIR=$3 # Optional: Path to a central folder for venvs
BASE_PATH=$(pwd)
FULL_PATH="$BASE_PATH/$TARGET_DIR"

# Safety Check: Abort if inside a Git Repo ---
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: You are currently inside a git repository."
    echo "Please run this script from a clean parent directory."
    exit 1
fi

# Input Validation ---
if [ -z "$BRANCH_NAME" ]; then
    echo "Usage: $0 <branch_name> [target_directory] [venv_base_path]"
    exit 1
fi

# Determine Venv Path ---
if [ -n "$VENV_BASE_DIR" ]; then
    # Ensure the base directory exists
    mkdir -p "$VENV_BASE_DIR"
    # Create a path based on the branch name
    VENV_PATH="$VENV_BASE_DIR/venv_$BRANCH_NAME"
else
    # Default to local .venv
    VENV_PATH="$FULL_PATH/.venv"
fi

echo "Cloning and Checking out Branch: $BRANCH_NAME ---"
git clone git@github.com:terrastackai/terratorch.git "$TARGET_DIR"
cd "$TARGET_DIR" || exit
git checkout "$BRANCH_NAME"

echo "Setting up Virtual Environment at: $VENV_PATH ---"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Installing Terratorch with Test Dependencies ---"
pip install --upgrade pip
pip install -e ".[test]"

echo "Submitting to LSF (bsub) ---"
# Note: Using VENV_PATH here so the remote job knows exactly where to look
bsub -gpu "num=1" -Is -R "rusage[ngpus=1, cpu=4, mem=128GB]" \
     -J "terratorch_ci_$BRANCH_NAME" \
     "cd $FULL_PATH && source $VENV_PATH/bin/activate && pytest ./tests"
