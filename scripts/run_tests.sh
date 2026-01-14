#!/bin/bash

BRANCH_NAME=$1
TARGET_DIR=${2:-"terratorch.$BRANCH_NAME"}
BASE_PATH=$(pwd)
FULL_PATH="$BASE_PATH/$TARGET_DIR"

# Safety Check: Abort if inside a Git Repo ---
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: You are currently inside a git repository."
    echo "Please run this script from a clean parent directory to avoid nested repositories."
    exit 1
fi

# Input Validation ---
if [ -z "$BRANCH_NAME" ]; then
    echo "Error: Please provide a branch name."
    echo "Usage: $0 <branch_name> [target_directory]"
    exit 1
fi

echo "Cloning and Checking out Branch: $BRANCH_NAME ---"
git clone git@github.com:terrastackai/terratorch.git "$TARGET_DIR"
cd "$TARGET_DIR" || exit
git checkout "$BRANCH_NAME"

echo "Setting up Virtual Environment ---"
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Terratorch with Test Dependencies ---"
pip install --upgrade pip
pip install -e ".[test]"

echo "Submitting to LSF (bsub) ---"
bsub -gpu "num=1" -Is -R "rusage[ngpus=1, cpu=4, mem=128GB]" \
     -J "terratorch_ci_$BRANCH_NAME" \
     "cd $FULL_PATH && source .venv/bin/activate && pytest ./tests"
