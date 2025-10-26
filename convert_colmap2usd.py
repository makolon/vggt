#!/usr/bin/env python3
"""
Convert COLMAP reconstruction output to USD format with physics properties.

This script reads COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
and converts the 3D point cloud to a USD file with collision and physics properties.

Usage:
    python convert_colmap2usd.py --scene_dir /path/to/scene --use_mesh
"""

import argparse
import sys
import numpy as np
import trimesh
from pathlib import Path

import pycolmap
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, UsdShade, Vt
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP reconstruction to USD")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Directory containing COLMAP sparse reconstruction (sparse/ folder)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output USD file path (default: scene_dir/sparse/scene.usd)"
    )
    parser.add_argument(
        "--use_mesh",
        action="store_true",
        help="Generate mesh from point cloud using Poisson reconstruction"
    )
    parser.add_argument(
        "--mesh_depth",
        type=int,
        default=9,
        help="Depth parameter for Poisson reconstruction (default: 9)"
    )
    parser.add_argument(
        "--collision_approximation",
        type=str,
        default="meshSimplification",
        choices=["convexDecomposition", "convexHull", "meshSimplification", "none"],
        help="Collision approximation method"
    )
    parser.add_argument(
        "--static_friction",
        type=float,
        default=0.5,
        help="Static friction coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--dynamic_friction",
        type=float,
        default=0.5,
        help="Dynamic friction coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--restitution",
        type=float,
        default=0.0,
        help="Restitution coefficient (default: 0.0)"
    )
    return parser.parse_args()


def load_colmap_reconstruction(sparse_dir):
    """Load COLMAP reconstruction from sparse directory."""
    print(f"Loading COLMAP reconstruction from {sparse_dir}...")
    
    # Try to load binary format first
    try:
        reconstruction = pycolmap.Reconstruction(sparse_dir)
        print(f"Loaded {len(reconstruction.points3D)} 3D points")
        print(f"Loaded {len(reconstruction.images)} images")
        print(f"Loaded {len(reconstruction.cameras)} cameras")
        return reconstruction
    except Exception as e:
        print(f"Error loading COLMAP reconstruction: {e}")
        sys.exit(1)


def load_point_cloud_from_ply(ply_path):
    """Load point cloud from PLY file as fallback."""
    print(f"Loading point cloud from {ply_path}...")
    mesh = trimesh.load(ply_path)
    if isinstance(mesh, trimesh.PointCloud):
        points = mesh.vertices
        colors = mesh.colors[:, :3] if mesh.colors is not None else None
    else:
        points = mesh.vertices
        colors = mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else None
    
    print(f"Loaded {len(points)} points from PLY")
    return points, colors


def reconstruction_to_point_cloud(reconstruction):
    """Convert COLMAP reconstruction to point cloud arrays."""
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        colors.append(point3D.color)
    
    points = np.array(points)
    colors = np.array(colors)
    
    return points, colors


def create_mesh_from_points(points, colors, depth=9):
    """Create mesh from point cloud using Poisson reconstruction."""
    print(f"Creating mesh from {len(points)} points using Poisson reconstruction...")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Normalize colors to [0, 1]
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    
    # Poisson reconstruction
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    
    # Remove low density vertices
    print("Cleaning mesh...")
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Convert to trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    
    if vertex_colors is not None:
        vertex_colors = (vertex_colors * 255).astype(np.uint8)
    
    trimesh_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    
    print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
    return trimesh_mesh


def create_usd_from_point_cloud(points, colors, output_path, args):
    """Create USD file from point cloud."""
    print(f"Creating USD file: {output_path}")
    
    # Create USD stage
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Create root prim
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)
    
    # Create geometry prim
    geom_path = "/World/Geometry"
    stage.DefinePrim(geom_path, "Xform")
    
    # Create point instancer
    points_path = f"{geom_path}/PointCloud"
    points_prim = UsdGeom.PointInstancer.Define(stage, points_path)
    
    # Set point positions
    points_np = np.asarray(points, dtype=np.float32)
    points_prim.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(points_np))
    
    # Set point colors if available
    if colors is not None:
        colors_normalized = colors.astype(np.float32) / 255.0
        primvars_api = UsdGeom.PrimvarsAPI(points_prim.GetPrim())
        primvar = primvars_api.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray
        )
        primvar.Set(Vt.Vec3fArray.FromNumpy(colors_normalized))
    
    # Set point IDs
    num_points = len(points)
    points_prim.GetProtoIndicesAttr().Set([0] * num_points)
    points_prim.GetIdsAttr().Set(list(range(num_points)))
    
    # Create a small sphere as prototype
    proto_path = f"{points_path}/Prototypes/Sphere"
    proto_sphere = UsdGeom.Sphere.Define(stage, proto_path)
    proto_sphere.GetRadiusAttr().Set(0.01)
    
    # Set prototypes
    points_prim.GetPrototypesRel().SetTargets([proto_path])
    
    stage.Save()
    print(f"USD file saved: {output_path}")


def create_usd_from_mesh(mesh, output_path, args):
    """Create USD file from mesh with physics properties."""
    print(f"Creating USD file with mesh: {output_path}")
    
    # Create USD stage
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    
    # Create root prim
    root_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(root_prim)
    
    # Create geometry prim
    geom_path = "/World/Geometry"
    geom_prim = stage.DefinePrim(geom_path, "Xform")
    
    # Create mesh
    mesh_path = f"{geom_path}/Mesh"
    usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Set mesh data
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Convert to float32 for USD compatibility
    vertices_np = np.asarray(vertices, dtype=np.float32)
    
    # USD expects flattened face indices
    face_vertex_counts = [3] * len(faces)
    face_vertex_indices = faces.flatten()
    
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices_np))
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices.tolist()))
    
    # Set vertex colors if available
    if mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        primvars_api = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim())
        primvar = primvars_api.CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec3fArray.FromNumpy(colors))
    
    # Add collision properties
    if args.collision_approximation != "none":
        collision_api = UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
        collision_api.CreateCollisionEnabledAttr().Set(True)
        
        # Set collision approximation
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim())
        if args.collision_approximation == "convexHull":
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")
        elif args.collision_approximation == "convexDecomposition":
            mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")
        elif args.collision_approximation == "meshSimplification":
            mesh_collision_api.CreateApproximationAttr().Set("meshSimplification")
    
    # Create physics material
    material_path = "/World/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(args.static_friction)
    physics_material.CreateDynamicFrictionAttr().Set(args.dynamic_friction)
    physics_material.CreateRestitutionAttr().Set(args.restitution)
    
    # Bind material to geometry
    binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim)
    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    
    stage.Save()
    print(f"USD file with physics properties saved: {output_path}")


def main():
    args = parse_args()
    
    # Setup paths
    scene_dir = Path(args.scene_dir)
    sparse_dir = scene_dir / "sparse"
    
    if not sparse_dir.exists():
        print(f"Error: Sparse directory not found: {sparse_dir}")
        sys.exit(1)
    
    # Determine output path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = sparse_dir / "scene.usd"
    
    # Load COLMAP reconstruction
    reconstruction = load_colmap_reconstruction(str(sparse_dir))
    
    # Get point cloud
    if len(reconstruction.points3D) > 0:
        points, colors = reconstruction_to_point_cloud(reconstruction)
    else:
        # Try to load from PLY as fallback
        ply_path = sparse_dir / "points.ply"
        if ply_path.exists():
            points, colors = load_point_cloud_from_ply(str(ply_path))
        else:
            print("Error: No 3D points found in reconstruction")
            sys.exit(1)
    
    if points is None or len(points) == 0:
        print("Error: No valid point cloud data")
        sys.exit(1)
    
    print("Point cloud statistics:")
    print(f"Number of points: {len(points)}")
    print(f"Bounding box min: {points.min(axis=0)}")
    print(f"Bounding box max: {points.max(axis=0)}")

    # Create USD file
    if args.use_mesh:        
        mesh = create_mesh_from_points(points, colors, depth=args.mesh_depth)
        if mesh is not None:
            create_usd_from_mesh(mesh, output_path, args)
            
            # Also save mesh as OBJ for reference
            obj_path = output_path.with_suffix('.obj')
            mesh.export(str(obj_path))
            print(f"Mesh also saved as OBJ: {obj_path}")
        else:
            print("Error: Failed to create mesh")
            sys.exit(1)
    else:
        create_usd_from_point_cloud(points, colors, output_path, args)
    
    print("\nConversion complete!")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
