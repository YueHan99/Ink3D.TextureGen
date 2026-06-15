#!/bin/bash
# Batch inference: v120 via H-model, shuffled order
set -e
cd /home/v-hanyue/workspace/Ink3D/OrbitVideoGen
export PYTHONPATH="/home/v-hanyue/workspace/Ink3D/OrbitVideoGen"
OUTDIR="./outputs_v_as_h"
mkdir -p "$OUTDIR"

# All cases with v120 data (excluding 001 already done), shuffled
CASES=(134 042 113 065 095 038 003 135 123 034 103 067 129 115)

for c in "${CASES[@]}"; do
  OUT="$OUTDIR/output_v_as_h_${c}.mp4"
  if [ -f "$OUT" ]; then
    echo "SKIP $c (already exists)"
    continue
  fi
  echo "=== Processing $c ==="
  python tests/test_single_v_as_h.py \
    --ref_image "tests/example_data/${c}/ref.png" \
    --video_dir "tests/example_data/${c}/v120/" \
    --output "$OUT" \
    2>&1 | tail -5
  echo "=== Done $c ==="
done
echo "ALL DONE"
