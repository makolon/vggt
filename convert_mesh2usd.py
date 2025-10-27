#!/usr/bin/env python3
"""
Convert mesh file (PLY/OBJ) to USD format with physics properties.

This script reads a mesh file (typically scene_dense.ply from OpenMVS)
and converts it to USD format with optional physics properties.

Usage:
    python convert_mesh2usd.py --input mesh/mvs/scene_dense.ply
    python convert_mesh2usd.py --input mesh/mvs/scene_dense.ply --output scene.usd
    python convert_mesh2usd.py --input mesh/mvs/scene_dense_mesh_refine.ply --export_obj
"""

import argparse
import sys
import numpy as np
import trimesh
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, UsdShade, Vt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert mesh file to USD/OBJ format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input mesh file (PLY, OBJ, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: same as input with .usd or .obj extension)"
    )
    parser.add_argument(
        "--export_obj",
        action="store_true",
        help="Export to OBJ format instead of USD"
    )
    parser.add_argument(
        "--export_both",
        action="store_true",
        help="Export both USD and OBJ formats"
    )
    parser.add_argument(
        "--collision_approximation",
        type=str,
        default="meshSimplification",
        choices=["convexDecomposition", "convexHull", "meshSimplification", "none"],
        help="Collision approximation method for USD (default: meshSimplification)"
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
    parser.add_argument(
        "--simplify",
        type=int,
        default=None,
        help="Simplify mesh to target number of faces (optional)"
    )
    parser.add_argument(
        "--up_axis",
        type=str,
        default="Z",
        choices=["X", "Y", "Z"],
        help="Up axis for USD (default: Z)"
    )
    return parser.parse_args()


def load_mesh(input_path):
    """Load mesh from file."""
    print(f"Loading mesh from: {input_path}")
    
    try:
        mesh = trimesh.load(input_path)
        
        # Handle scene objects (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            print(f"Loaded scene with {len(mesh.geometry)} geometries")
            # Combine all geometries into single mesh
            meshes = list(mesh.geometry.values())
            if len(meshes) == 0:
                print("Error: No geometries found in scene")
                return None
            elif len(meshes) == 1:
                mesh = meshes[0]
            else:
                mesh = trimesh.util.concatenate(meshes)
                print(f"Combined {len(meshes)} meshes into one")
        
        print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Print mesh statistics
        print("\nMesh statistics:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Is watertight: {mesh.is_watertight}")
        print(f"  Bounding box min: {mesh.bounds[0]}")
        print(f"  Bounding box max: {mesh.bounds[1]}")
        print(f"  Extents: {mesh.extents}")
        
        if mesh.visual.kind == 'vertex':
            print("Has vertex colors: Yes")
        elif mesh.visual.kind == 'texture':
            print("Has texture: Yes")
        
        return mesh
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def simplify_mesh(mesh, target_faces):
    """Simplify mesh to target number of faces."""
    print(f"\nSimplifying mesh from {len(mesh.faces)} to ~{target_faces} faces...")
    
    try:
        # Use quadric decimation for better quality
        simplified = mesh.simplify_quadric_decimation(target_faces)
        print(f"Simplified to {len(simplified.faces)} faces ({len(simplified.vertices)} vertices)")
        return simplified
    except Exception as e:
        print(f"Error simplifying mesh: {e}")
        print("Continuing with original mesh...")
        return mesh


def export_obj(mesh, output_path):
    """Export mesh to OBJ format."""
    print(f"\nExporting to OBJ: {output_path}")
    
    try:
        # Export mesh
        mesh.export(str(output_path))
        print(f"Successfully exported OBJ: {output_path}")
        
        # Check for MTL and texture files
        mtl_path = output_path.with_suffix('.mtl')
        if mtl_path.exists():
            print(f"  Material file: {mtl_path}")
        
        # List any texture files in the same directory
        texture_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            texture_files.extend(output_path.parent.glob(f"*{ext}"))
        
        if texture_files:
            print(f"  Found {len(texture_files)} texture files in directory")
        
        return True
        
    except Exception as e:
        print(f"Error exporting OBJ: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_usd(mesh, output_path, args):
    """Export mesh to USD format with physics properties."""
    print(f"\nExporting to USD: {output_path}")
    
    try:
        # Create USD stage
        stage = Usd.Stage.CreateNew(str(output_path))
        
        # Set up axis
        if args.up_axis == "Z":
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        elif args.up_axis == "Y":
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        else:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.x)
        
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        # Add physics if collision is enabled
        if args.collision_approximation != "none":
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
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces
        
        # USD expects flattened face indices
        face_vertex_counts = [len(face) for face in faces]
        face_vertex_indices = faces.flatten()
        
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices.tolist()))
        
        # Set vertex colors if available
        if mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            primvars_api = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim())
            primvar = primvars_api.CreatePrimvar(
                "displayColor",
                Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.vertex
            )
            primvar.Set(Vt.Vec3fArray.FromNumpy(colors))
            print("  Added vertex colors")
        
        # Handle texture materials
        if mesh.visual.kind == 'texture' and hasattr(mesh.visual, 'material'):
            print("  Note: Texture materials detected but not yet implemented for USD export")
            # TODO: Implement texture material export
        
        # Add collision properties
        if args.collision_approximation != "none":
            print(f"  Adding collision: {args.collision_approximation}")
            collision_api = UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
            collision_api.CreateCollisionEnabledAttr().Set(True)
            
            # Set collision approximation
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim())
            mesh_collision_api.CreateApproximationAttr().Set(args.collision_approximation)
        
        # Create physics material
        if args.collision_approximation != "none":
            material_path = "/World/PhysicsMaterial"
            material = UsdShade.Material.Define(stage, material_path)
            physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            physics_material.CreateStaticFrictionAttr().Set(args.static_friction)
            physics_material.CreateDynamicFrictionAttr().Set(args.dynamic_friction)
            physics_material.CreateRestitutionAttr().Set(args.restitution)
            
            # Bind material to geometry
            binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            
            print(f"  Physics material: friction={args.static_friction}, restitution={args.restitution}")
        
        # Save stage
        stage.Save()
        print(f"Successfully exported USD: {output_path}")
        
        # Print file info
        file_size = output_path.stat().st_size
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error exporting USD: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output paths
    if args.output:
        output_base = Path(args.output).with_suffix('')
    else:
        output_base = input_path.with_suffix('')
    
    usd_output = output_base.with_suffix('.usd')
    obj_output = output_base.with_suffix('.obj')
    
    print("="*80)
    print("Mesh to USD/OBJ Converter")
    print("="*80)
    print(f"Input: {input_path}")
    
    if args.export_both:
        print(f"Output USD: {usd_output}")
        print(f"Output OBJ: {obj_output}")
    elif args.export_obj:
        print(f"Output OBJ: {obj_output}")
    else:
        print(f"Output USD: {usd_output}")
    
    print("="*80)
    
    # Load mesh
    mesh = load_mesh(input_path)
    if mesh is None:
        sys.exit(1)
    
    # Simplify if requested
    if args.simplify is not None:
        mesh = simplify_mesh(mesh, args.simplify)
    
    # Export
    success = True
    
    if args.export_both:
        # Export both formats
        success_obj = export_obj(mesh, obj_output)
        success_usd = export_usd(mesh, usd_output, args)
        success = success_obj or success_usd
    elif args.export_obj:
        # Export OBJ only
        success = export_obj(mesh, obj_output)
    else:
        # Export USD only
        success = export_usd(mesh, usd_output, args)
    
    if success:
        print("\n" + "="*80)
        print("Conversion completed successfully!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("Conversion failed!")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
