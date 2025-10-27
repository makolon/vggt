#!/usr/bin/env python3
"""
Convert COLMAP sparse reconstruction to mesh using OpenMVS.

This script takes COLMAP sparse reconstruction output and generates a textured mesh
using OpenMVS (Open Multi-View Stereo) pipeline.

Usage:
    python convert_colmap2mesh.py --scene_dir /path/to/scene
    python convert_colmap2mesh.py --scene_dir /path/to/scene --no_texture
    python convert_colmap2mesh.py --scene_dir /path/to/scene --max_face_area 32

Requirements:
    - OpenMVS installed and available in PATH (InterfaceCOLMAP, DensifyPointCloud, etc.)
    - COLMAP installed (for image_undistorter)
"""

import argparse
import subprocess
import os
import sys
import json
import shutil
from pathlib import Path

import pycolmap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COLMAP sparse reconstruction to mesh using OpenMVS"
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing COLMAP sparse reconstruction"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Skip texture mapping (faster but no color)"
    )
    parser.add_argument(
        "--max_face_area",
        type=int,
        default=16,
        help="Maximum face area for mesh refinement (default: 16)"
    )
    parser.add_argument(
        "--refine_scales",
        type=str,
        default="1",
        help="Scales for mesh refinement (default: '1')"
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip processing if output already exists"
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate files (dense folder, etc.)"
    )
    return parser.parse_args()


def check_dependencies():
    """Check if required external commands are available."""
    required_commands = [
        "colmap",
        "InterfaceCOLMAP",
        "DensifyPointCloud",
        "ReconstructMesh",
        "RefineMesh",
        "TextureMesh"
    ]
    
    missing = []
    for cmd in required_commands:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    if missing:
        print("Error: The following required commands are not found in PATH:")
        for cmd in missing:
            print(f"  - {cmd}")
        print("\nPlease install OpenMVS and COLMAP and ensure they are in your PATH.")
        sys.exit(1)


def run_command(cmd, cwd=None, description=""):
    """Run a shell command and handle errors."""
    if description:
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {cwd if cwd else os.getcwd()}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print(result.stdout)
    return True


def validate_input_structure(scene_dir):
    """Validate that the input directory has the expected structure."""
    scene_path = Path(scene_dir)
    
    # Check images directory
    images_dir = scene_path / "images"
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Count images
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    if len(image_files) == 0:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Check sparse reconstruction
    sparse_dir = scene_path / "sparse"
    if not sparse_dir.exists():
        print(f"Error: Sparse directory not found: {sparse_dir}")
        sys.exit(1)
    
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for file in required_files:
        file_path = sparse_dir / file
        if not file_path.exists():
            print(f"Error: Required file not found: {file_path}")
            sys.exit(1)
    
    print(f"Validated sparse reconstruction in {sparse_dir}")
    return True


def _camera_model_name(camera) -> str:
    if hasattr(camera, "model") and hasattr(camera.model, "name"):
        return camera.model.name
    if hasattr(camera, "model_name"):
        return camera.model_name
    return str(getattr(camera, "model", ""))


def convert_cameras_to_pinhole(sparse_dir, output_dir):
    """Convert camera models to PINHOLE format for OpenMVS."""
    print("Converting cameras to PINHOLE model...")
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))

    for cam_id, camera in reconstruction.cameras.items():
        model_name = _camera_model_name(camera)

        if model_name != "PINHOLE":
            print(f"  Converting camera {cam_id} from {model_name} to PINHOLE")

            width = int(camera.width)
            height = int(camera.height)
            params = list(camera.params)

            if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
                # f, cx, cy
                f, cx, cy = params[0], params[1], params[2]
                fx = fy = f
            elif model_name in ("PINHOLE", "RADIAL", "RADIAL_FISHEYE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                print(f"  Warning: Unknown camera model {model_name}, using fallback intrinsics")
                fx = fy = max(width, height)
                cx = width / 2.0
                cy = height / 2.0

            new_camera = pycolmap.Camera(
                model="PINHOLE",
                width=width,
                height=height,
                params=[float(fx), float(fy), float(cx), float(cy)],
                camera_id=int(cam_id),
            )
            reconstruction.cameras[cam_id] = new_camera
        else:
            print(f"  Camera {cam_id} is already PINHOLE model")

    os.makedirs(output_dir, exist_ok=True)
    reconstruction.write(str(output_dir))
    print(f"  Saved PINHOLE cameras to {output_dir}")
    return True


def step_undistort_images(scene_dir, dense_dir):
    """Step 1: Prepare images and convert to PINHOLE model."""
    print("\n" + "="*80)
    print("STEP 1: Preparing images and converting cameras to PINHOLE")
    print("="*80)
    
    scene_path = Path(scene_dir)
    images_dir = scene_path / "images"
    sparse_dir = scene_path / "sparse"
    
    # Create dense output directory structure
    os.makedirs(dense_dir, exist_ok=True)
    dense_images_dir = Path(dense_dir) / "images"
    dense_sparse_dir = Path(dense_dir) / "sparse"
    
    # Instead of undistorting, we'll work with original images
    # This avoids dimension mismatch issues
    print("Creating symlink to original images...")
    if dense_images_dir.exists():
        if dense_images_dir.is_symlink():
            dense_images_dir.unlink()
        else:
            shutil.rmtree(dense_images_dir)
    
    # Create symlink to images
    dense_images_dir.symlink_to(images_dir.resolve())
    print(f"  Linked {dense_images_dir} -> {images_dir}")
    
    # Convert cameras to PINHOLE model in the dense/sparse directory
    print("Converting cameras to PINHOLE model...")
    if not convert_cameras_to_pinhole(sparse_dir, dense_sparse_dir):
        print("Warning: Could not convert cameras to PINHOLE model, copying original...")
        shutil.copytree(sparse_dir, dense_sparse_dir, dirs_exist_ok=True)
    
    # Verify output structure
    if not dense_images_dir.exists():
        print(f"Error: Images not accessible at {dense_images_dir}")
        sys.exit(1)
    
    if not dense_sparse_dir.exists():
        print(f"Error: Sparse reconstruction not found at {dense_sparse_dir}")
        sys.exit(1)
    
    print(f"Images and cameras prepared successfully in {dense_dir}")
    return True


def step_interface_colmap(dense_dir, mvs_dir):
    """Step 2: Convert COLMAP to MVS format."""
    print("\n" + "="*80)
    print("STEP 2: Converting COLMAP to MVS format")
    print("="*80)
    
    os.makedirs(mvs_dir, exist_ok=True)
    
    scene_mvs = os.path.join(mvs_dir, "scene.mvs")
    images_dir = os.path.join(dense_dir, "images")
    
    cmd = [
        "InterfaceCOLMAP",
        "-i", str(dense_dir),
        "-o", scene_mvs,
        "--image-folder", images_dir
    ]
    
    success = run_command(cmd, cwd=mvs_dir, description="Converting to MVS format")
    
    if not success or not os.path.exists(scene_mvs):
        print("Error: COLMAP to MVS conversion failed")
        sys.exit(1)
    
    print(f"MVS scene file created: {scene_mvs}")
    return scene_mvs


def step_densify_point_cloud(mvs_dir, scene_mvs):
    """Step 3: Densify point cloud."""
    print("\n" + "="*80)
    print("STEP 3: Densifying point cloud")
    print("="*80)
    
    cmd = [
        "DensifyPointCloud",
        "scene.mvs"
    ]
    
    success = run_command(cmd, cwd=mvs_dir, description="Densifying point cloud")
    
    dense_mvs = os.path.join(mvs_dir, "scene_dense.mvs")
    if not success or not os.path.exists(dense_mvs):
        print("Error: Point cloud densification failed")
        sys.exit(1)
    
    print(f"Dense point cloud created: {dense_mvs}")
    return dense_mvs


def step_reconstruct_mesh(mvs_dir):
    """Step 4: Reconstruct mesh from dense point cloud."""
    print("\n" + "="*80)
    print("STEP 4: Reconstructing mesh")
    print("="*80)
    
    cmd = [
        "ReconstructMesh",
        "scene_dense.mvs",
        "-p", "scene_dense.ply"
    ]
    
    success = run_command(cmd, cwd=mvs_dir, description="Reconstructing mesh")
    
    mesh_ply = os.path.join(mvs_dir, "scene_dense_mesh.ply")
    if not success or not os.path.exists(mesh_ply):
        print("Error: Mesh reconstruction failed")
        sys.exit(1)
    
    print(f"Mesh reconstructed: {mesh_ply}")
    return mesh_ply


def step_refine_mesh(mvs_dir, max_face_area, refine_scales):
    """Step 5: Refine mesh."""
    print("\n" + "="*80)
    print("STEP 5: Refining mesh")
    print("="*80)
    
    cmd = [
        "RefineMesh",
        "scene_dense.mvs",
        "-m", "scene_dense_mesh.ply",
        "-o", "scene_dense_mesh_refine.mvs",
        "--scales", refine_scales,
        "--max-face-area", str(max_face_area)
    ]
    
    success = run_command(cmd, cwd=mvs_dir, description="Refining mesh")
    
    refined_mesh = os.path.join(mvs_dir, "scene_dense_mesh_refine.ply")
    if not success or not os.path.exists(refined_mesh):
        print("Error: Mesh refinement failed")
        sys.exit(1)
    
    print(f"Mesh refined: {refined_mesh}")
    return refined_mesh


def step_texture_mesh(mvs_dir):
    """Step 6: Apply texture to mesh."""
    print("\n" + "="*80)
    print("STEP 6: Applying texture to mesh")
    print("="*80)
    
    cmd = [
        "TextureMesh",
        "scene_dense.mvs",
        "-m", "scene_dense_mesh_refine.ply",
        "-o", "scene_dense_mesh_refine_texture.mvs"
    ]
    
    success = run_command(cmd, cwd=mvs_dir, description="Texturing mesh")
    
    textured_mesh = os.path.join(mvs_dir, "scene_dense_mesh_refine_texture.ply")
    if not success or not os.path.exists(textured_mesh):
        print("Error: Mesh texturing failed")
        sys.exit(1)
    
    print(f"Textured mesh created: {textured_mesh}")
    return textured_mesh


def create_progress_file(mvs_dir, args):
    """Create a progress file to track completion."""
    progress = {
        "completed": True,
        "scene_dir": str(args.scene_dir),
        "max_face_area": args.max_face_area,
        "refine_scales": args.refine_scales,
        "textured": not args.no_texture
    }
    
    progress_file = os.path.join(mvs_dir, "progress.json")
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)
    
    print(f"\nProgress file saved: {progress_file}")


def cleanup_intermediate_files(dense_dir, keep_intermediate):
    """Clean up intermediate files if requested."""
    if keep_intermediate:
        print("\nKeeping intermediate files")
        return
    
    print("\n" + "="*80)
    print("Cleaning up intermediate files")
    print("="*80)
    
    if os.path.exists(dense_dir):
        print(f"Removing dense directory: {dense_dir}")
        shutil.rmtree(dense_dir)
        print("Intermediate files cleaned up")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("COLMAP to Mesh Conversion using OpenMVS")
    print("="*80)
    print(f"Scene directory: {args.scene_dir}")
    print(f"Max face area: {args.max_face_area}")
    print(f"Refine scales: {args.refine_scales}")
    print(f"Texture: {'No' if args.no_texture else 'Yes'}")
    print("="*80)
    
    # Check dependencies
    check_dependencies()
    
    # Setup paths
    scene_path = Path(args.scene_dir).resolve()
    dense_dir = scene_path / "dense"
    mesh_dir = scene_path / "mesh"
    mvs_dir = mesh_dir / "mvs"
    
    # Check if output already exists
    final_mesh = mvs_dir / ("scene_dense_mesh_refine_texture.ply" if not args.no_texture else "scene_dense_mesh_refine.ply")
    if args.skip_if_exists and final_mesh.exists():
        print(f"\nOutput already exists: {final_mesh}")
        print("Skipping processing (use --no-skip-if-exists to force reprocessing)")
        return
    
    # Validate input
    validate_input_structure(args.scene_dir)
    
    # Pipeline execution
    # Step 1: Prepare images (skip undistortion to avoid dimension mismatches)
    step_undistort_images(args.scene_dir, dense_dir)
    
    # Step 2: Convert to MVS format
    scene_mvs = step_interface_colmap(dense_dir, mvs_dir)
    
    # Step 3: Densify point cloud
    step_densify_point_cloud(mvs_dir, scene_mvs)
    
    # Step 4: Reconstruct mesh
    step_reconstruct_mesh(mvs_dir)
    
    # Step 5: Refine mesh
    refined_mesh = step_refine_mesh(mvs_dir, args.max_face_area, args.refine_scales)
    
    # Step 6: Texture mesh (optional)
    if not args.no_texture:
        textured_mesh = step_texture_mesh(mvs_dir)
        final_output = textured_mesh
    else:
        final_output = refined_mesh
        print("\nSkipping texture mapping (--no_texture flag set)")
    
    # Create progress file
    create_progress_file(mvs_dir, args)
    
    # Cleanup
    cleanup_intermediate_files(dense_dir, args.keep_intermediate)
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {mesh_dir}")
    print(f"Final mesh: {final_output}")
    
    if not args.no_texture:
        print("\nTextured mesh files:")
        print(f"- PLY: {final_output}")
        texture_files = list(Path(mvs_dir).glob("scene_dense_mesh_refine_texture*.png"))
        if texture_files:
            print(f"- Textures: {len(texture_files)} files")
    else:
        print(f"\nUntextured mesh: {final_output}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
