#!/bin/bash

# --- Configuration ---
# Usage: ./run_tests.sh <branch_name> [target_directory]
BRANCH_NAME=$1
TARGET_DIR=${2:-"terratorch.$BRANCH_NAME"}
BASE_PATH=$(pwd)
FULL_PATH="$BASE_PATH/$TARGET_DIR"

if [ -z "$BRANCH_NAME" ]; then
    echo "Error: Please provide a branch name."
    echo "Usage: $0 <branch_name> [target_directory]"
    exit 1
fi

echo "--- 1. Cloning and Checking out Branch: $BRANCH_NAME ---"
git clone git@github.com:terrastackai/terratorch.git "$TARGET_DIR"
cd "$TARGET_DIR" || exit
git checkout "$BRANCH_NAME"

echo "--- 2. Setting up Virtual Environment ---"
python3 -m venv .venv
source .venv/bin/activate

echo "--- 3. Installing Terratorch with Test Dependencies ---"
pip install --upgrade pip
pip install -e ".[test]"

echo "--- 4. Running Local Pytest (Sanity Check) ---"
pytest tests

echo "--- 5. Submitting to LSF (bsub) ---"
# This command sends the job to the cluster
bsub -gpu "num=1" -Is -R "rusage[ngpus=1, cpu=4, mem=128GB]" \
     -J "terratorch_ci_$BRANCH_NAME" \
     "cd $FULL_PATH && source .venv/bin/activate && pytest ./tests"
