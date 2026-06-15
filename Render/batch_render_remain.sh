#!/bin/bash
# Complete remaining re-renders
set -e
BASE="/home/v-hanyue/workspace/Ink3D/OrbitVideoGen/tests/example_data"
RENDER="/home/v-hanyue/workspace/Ink3D/Render/render.py"
ENV="bpy40"

# V only (H is already 120)
for case in 052 126 gir5; do
    echo "===== $case (V only) ====="
    GLB="$BASE/$case/mesh.glb"
    [ ! -f "$GLB" ] && echo "SKIP: no mesh.glb" && continue
    rm -rf "$BASE/$case/v120"
    conda run -n $ENV python3 $RENDER \
        --input_file "$GLB" --output_dir "$BASE" \
        --orbit vertical --num_cameras 120 --model_name "$case" --flip_x
done

# H+V (both need re-render)
for case in mingren piano piano2 rabbit spider tent toy8; do
    echo "===== $case (H+V) ====="
    GLB="$BASE/$case/mesh.glb"
    [ ! -f "$GLB" ] && echo "SKIP: no mesh.glb" && continue
    rm -rf "$BASE/$case/h120" "$BASE/$case/v120"
    conda run -n $ENV python3 $RENDER \
        --input_file "$GLB" --output_dir "$BASE" \
        --orbit horizontal --num_cameras 120 --model_name "$case"
    conda run -n $ENV python3 $RENDER \
        --input_file "$GLB" --output_dir "$BASE" \
        --orbit vertical --num_cameras 120 --model_name "$case" --flip_x
done

echo "ALL DONE"
