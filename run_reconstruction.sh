#!/bin/bash
# VGGT + OpenMVS Reconstruction Pipeline
# This script runs the complete 3D reconstruction pipeline

set -e  # Exit on error

SCENE_DIR="${1:-./examples/table}"

echo "=================================="
echo "3D Reconstruction Pipeline"
echo "=================================="
echo "Scene directory: $SCENE_DIR"
echo ""

# Step 1: Clean previous sparse reconstruction
echo "Step 1: Cleaning previous reconstruction..."
if [ -d "$SCENE_DIR/sparse" ]; then
    echo "Removing existing sparse directory..."
    rm -rf "$SCENE_DIR/sparse"
fi

if [ -d "$SCENE_DIR/dense" ]; then
    echo "Removing existing dense directory..."
    rm -rf "$SCENE_DIR/dense"
fi

if [ -d "$SCENE_DIR/mesh" ]; then
    echo "Removing existing mesh directory..."
    rm -rf "$SCENE_DIR/mesh"
fi

echo ""

# Step 2: Run VGGT with Bundle Adjustment for better visibility
echo "Step 2: Running VGGT with Bundle Adjustment..."
echo "This will create sparse reconstruction with multi-view observations"
echo ""

python3 demo_colmap.py \
    --scene_dir "$SCENE_DIR" \
    --use_ba \
    --shared_camera \
    --vis_thresh 0.1 \
    --query_frame_num 25 \
    --max_query_pts 8192 \
    --fine_tracking \
    --max_reproj_error 4.0

echo ""
echo "Step 2 completed: Sparse reconstruction saved to $SCENE_DIR/sparse"
echo ""

# Step 3: Check reconstruction quality
echo "Step 3: Checking reconstruction quality..."
python3 -c "
import pycolmap
import sys

reconstruction = pycolmap.Reconstruction('$SCENE_DIR/sparse')
print(f'Images: {len(reconstruction.images)}')
print(f'Points: {len(reconstruction.points3D)}')
print(f'Cameras: {len(reconstruction.cameras)}')

# Check visibility
if len(reconstruction.points3D) > 0:
    vis_counts = {}
    for point3D_id, point3D in reconstruction.points3D.items():
        track_length = len(point3D.track.elements)
        vis_counts[track_length] = vis_counts.get(track_length, 0) + 1
    
    print(f'\nVisibility statistics:')
    for track_len in sorted(vis_counts.keys()):
        count = vis_counts[track_len]
        percentage = 100.0 * count / len(reconstruction.points3D)
        print(f'  {count} points with {track_len} views ({percentage:.2f}%)')
    
    # Check if we have enough multi-view points
    multi_view_points = sum(count for track_len, count in vis_counts.items() if track_len >= 2)
    multi_view_ratio = multi_view_points / len(reconstruction.points3D)
    
    if multi_view_ratio < 0.1:
        print(f'\nWARNING: Only {multi_view_ratio*100:.1f}% of points are visible in 2+ views')
        print('   OpenMVS mesh refinement may fail. Consider:')
        print('   1. Using --skip_refine flag with convert_colmap2mesh.py')
        print('   2. Increasing --query_frame_num and --max_query_pts')
        print('   3. Using more overlapping images')
    else:
        print(f'\n✓ Good: {multi_view_ratio*100:.1f}% of points are visible in 2+ views')
else:
    print('\nWARNING: No 3D points in reconstruction!')
    sys.exit(1)
"

echo ""

# Step 4: Convert to mesh
echo "Step 4: Converting COLMAP sparse reconstruction to mesh..."
echo ""

python3 convert_colmap2mesh.py \
    --scene_dir "$SCENE_DIR" \
    --max_face_area 16 \
    --refine_scales 1

echo ""

# Step 5: Convert mesh to USD and OBJ formats
echo "Step 5: Converting mesh to USD and OBJ formats..."
echo ""

# Check which mesh file to convert (prefer textured version)
MESH_FILE=""
if [ -f "$SCENE_DIR/mesh/mvs/scene_dense_mesh_refine_texture.ply" ]; then
    MESH_FILE="$SCENE_DIR/mesh/mvs/scene_dense_mesh_refine_texture.ply"
    echo "Using textured mesh: $MESH_FILE"
elif [ -f "$SCENE_DIR/mesh/mvs/scene_dense_mesh_refine.ply" ]; then
    MESH_FILE="$SCENE_DIR/mesh/mvs/scene_dense_mesh_refine.ply"
    echo "Using refined mesh: $MESH_FILE"
elif [ -f "$SCENE_DIR/mesh/mvs/scene_dense.ply" ]; then
    MESH_FILE="$SCENE_DIR/mesh/mvs/scene_dense.ply"
    echo "Using dense point cloud: $MESH_FILE"
else
    echo "Warning: No mesh file found, skipping USD/OBJ conversion"
    MESH_FILE=""
fi

if [ -n "$MESH_FILE" ]; then
    # Export both USD and OBJ formats
    python3 convert_mesh2usd.py \
        --input "$MESH_FILE" \
        --export_both \
        --collision_approximation meshSimplification \
        --static_friction 0.5 \
        --dynamic_friction 0.5 \
        --restitution 0.0
    
    echo ""
    echo "Mesh converted to USD and OBJ formats"
fi

echo ""
echo "=================================="
echo "Pipeline completed successfully!"
echo "=================================="
echo ""
echo "Output files:"
echo "  - Sparse reconstruction: $SCENE_DIR/sparse/"
echo "  - Point cloud: $SCENE_DIR/sparse/points.ply"
echo "  - Mesh: $SCENE_DIR/mesh/mvs/"
if [ -n "$MESH_FILE" ]; then
    USD_FILE="${MESH_FILE%.ply}.usd"
    OBJ_FILE="${MESH_FILE%.ply}.obj"
    echo "  - USD file: $USD_FILE"
    echo "  - OBJ file: $OBJ_FILE"
fi
echo ""
