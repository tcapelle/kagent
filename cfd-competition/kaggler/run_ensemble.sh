#!/bin/bash
# Run ensemble prediction after training completes
set -e

echo "Running 20-checkpoint ensemble prediction..."
uv run python predict.py --checkpoint_list ensemble_checkpoints.txt --agent fern --top_k 20
echo "Done! Predictions saved."
