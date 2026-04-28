#!/usr/bin/env bash
# Eval 4-way per-split ensemble (v7+v10+v19+v21).
# Sweep weights for v21 from 0.0 (drops to 3-way) to 0.40, with v7/v10/v19 splitting the rest.

set -e
cd /workspace/kagent/tandemfoil-competition/kaggler

V7=models/model-izweqran/checkpoint.pt   # slice=64 chain v3 lr=1e-5
V10=models/model-8fazsiug/checkpoint.pt  # slice=128 chain v8 lr=1e-5
V19=models/model-bxvp8448/checkpoint.pt  # slice=32 chain v14 lr=5e-6
V21=$1                                     # passed in
[[ -z "$V21" ]] && { echo "usage: $0 <v21-ckpt-path>"; exit 1; }

# Build a list of 4-way configs varying v21 weight 0..0.4 step 0.05, and within
# each total budget for v21 vary v7 from 0.4..0.7, v10 0.2..0.4, v19 fills.
CFGS=()
for d in 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40; do
  remaining=$(python -c "print(round(1.0-$d, 4))")
  for a in 0.40 0.45 0.50 0.55 0.60 0.65 0.70; do
    for b in 0.15 0.20 0.25 0.30 0.35 0.40; do
      c=$(python -c "print(round($remaining-$a-$b, 4))")
      ok=$(python -c "print(1 if 0.0 <= $c <= 0.35 else 0)")
      [[ "$ok" == "1" ]] && CFGS+=("$a,$b,$c,$d")
    done
  done
done

echo "Evaluating ${#CFGS[@]} configs..."
python eval_per_split.py \
  --checkpoints "$V7" "$V10" "$V19" "$V21" \
  --weight_configs "${CFGS[@]}"
