#!/usr/bin/env bash
# Post-v22 evaluation: try 4-way ensembles with v22 alongside v7/v10/v19.
# Usage: ./run_post_v22.sh <v22_ckpt_path>

set -e
cd /workspace/kagent/tandemfoil-competition/kaggler

V7=models/model-izweqran/checkpoint.pt
V10=models/model-8fazsiug/checkpoint.pt
V19=models/model-bxvp8448/checkpoint.pt
V22="$1"
[[ -z "$V22" ]] && { echo "usage: $0 <v22-ckpt-path>"; exit 1; }

# Also include v3 (parent of v7) and v8 (parent of v10) and v6 (fresh slice=64) for completeness
V3=models/model-l0nw6exf/checkpoint.pt
V6=models/model-pnn6ips2/checkpoint.pt
V8=models/model-f3s5vkhf/checkpoint.pt
V14=models/model-mj69t5iv/checkpoint.pt

python eval_grid.py \
  --checkpoints "$V7" "$V10" "$V19" "$V22" "$V3" "$V6" "$V8" "$V14" \
  --subsets "0,1,2" "0,1,2,3" "0,1,2,4" "0,1,2,5" "0,1,2,6" "0,1,2,7" "0,1,2,3,5" "0,1,2,3,7" \
  --grid_step 0.05
