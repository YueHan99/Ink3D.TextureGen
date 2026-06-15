#!/bin/bash
# Batch re-render all 121-frame cases → 120 frames, H + V (with flip_x)
set -e

CASES="bag bottle building5 cake cart castle chef cloth gir5 mingren panda piano piano2 rabbit spider tent toy8"
BASE="/home/v-hanyue/workspace/Ink3D/OrbitVideoGen/tests/example_data"
RENDER_SCRIPT="/home/v-hanyue/workspace/Ink3D/Render/render.py"
ENV="bpy40"

for case in $CASES; do
    echo "===== $case ====="
    GLB="$BASE/$case/mesh.glb"
    if [ ! -f "$GLB" ]; then
        echo "SKIP: no mesh.glb"
        continue
    fi

    # H track
    echo "--- H ---"
    rm -rf "$BASE/$case/h120"
    conda run -n $ENV python3 $RENDER_SCRIPT \
        --input_file "$GLB" \
        --output_dir "$BASE" \
        --orbit horizontal \
        --num_cameras 120 \
        --model_name "$case"

    # V track (flip_x)
    echo "--- V ---"
    rm -rf "$BASE/$case/v120"
    conda run -n $ENV python3 $RENDER_SCRIPT \
        --input_file "$GLB" \
        --output_dir "$BASE" \
        --orbit vertical \
        --num_cameras 120 \
        --model_name "$case" \
        --flip_x

    echo "===== $case DONE ====="
done

echo "ALL DONE"
