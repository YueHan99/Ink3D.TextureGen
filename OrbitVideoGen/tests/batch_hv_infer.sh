#!/bin/bash
# Batch H+V inference for all ready cases
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"

CKPT_HIGH="${CKPT_HIGH:-}"
CKPT_LOW="${CKPT_LOW:-}"
BASE="tests/example_data"
OUTDIR="output/batch_hv"

mkdir -p "$OUTDIR"

CASES=$(echo "001 003 034 038 042 065 067 095 103 113 115 123 126 129 134 135 bag bottle building5 cake cart castle chef cloth gir5 mingren panda piano piano2" | tr ' ' '\n' | shuf)
for case in $CASES; do
    echo "===== $case ====="

    REF="$BASE/$case/ref.png"
    if [ ! -f "$REF" ]; then
        REF="$BASE/$case/G000.png"
    fi
    if [ ! -f "$REF" ]; then
        echo "SKIP: no ref image for $case"
        continue
    fi

    H_OUT="$OUTDIR/h_${case}.mp4"
    V_OUT="$OUTDIR/v_${case}.mp4"

    echo "--- H ---"
    python tests/test_single_h.py \
        --ref_image "$REF" \
        --video_dir "$BASE/$case/h120" \
        --model_ckpt_high "$CKPT_HIGH" \
        --model_ckpt_low "$CKPT_LOW" \
        --output "$H_OUT"

    echo "--- V ---"
    python tests/test_single_v_hv.py \
        --ref_image "$REF" \
        --video_dir "$BASE/$case/v120" \
        --h_video "$H_OUT" \
        --model_ckpt_high "$CKPT_HIGH" \
        --model_ckpt_low "$CKPT_LOW" \
        --output "$V_OUT"

    echo "===== $case DONE ====="
done

echo "ALL DONE"
