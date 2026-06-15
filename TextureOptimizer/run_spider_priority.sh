#!/bin/bash
# Priority baking for spider example
cd /home/v-hanyue/workspace/Ink3D/TextureOptimizer/

export BLENDER_PATH=/tmp/blender-4.5.1/blender-4.5.1-linux-x64/blender

# Step 1: Bake with priority mode
echo "=== Step 1: Priority baking ==="
/home/v-hanyue/miniconda3/envs/trellis2/bin/python3 voxelize.py \
    /home/v-hanyue/workspace/Ink3D/OrbitVideoGen/tests/example_data/spider/mesh.glb \
    --video test_outputs/spider_G000/h_spider_G000.mp4 \
    --video_v test_outputs/spider_G000/v_spider_G000.mp4 \
    --video_num_cols 4 --video_col 2 \
    --video_v_num_cols 4 --video_v_col 2 \
    --priority_mode \
    --output_vxz test_outputs/spider_G000/spider_priority.vxz \
    --resolution 1024

# Step 2: PBR render
echo "=== Step 2: PBR render ==="
TRELLIS_ROOT=/home/v-hanyue/blobmnt/workspace/TRELLIS.2.train.gen/TRELLIS.2
PYTHONPATH=$TRELLIS_ROOT:$PYTHONPATH /home/v-hanyue/miniconda3/envs/trellis2/bin/python3 render_vxz.py \
    --vxz test_outputs/spider_G000/spider_priority.vxz \
    --mesh test_outputs/spider_G000/spider_priority.pickle \
    -o test_outputs/spider_G000/spider_priority_pbr.mp4 \
    --roughness 0.15 --metallic 0.3 --turntable --shaded_only

echo "=== Done ==="
echo "Outputs:"
echo "  test_outputs/spider_G000/spider_priority.vxz"
echo "  test_outputs/spider_G000/spider_priority.pickle"
echo "  test_outputs/spider_G000/spider_priority_pbr.mp4"
