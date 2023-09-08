"""Process shapeNet data"""
import importlib_resources
import os
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R

asset_dir = importlib_resources.files('lgmcts.assets')
shapeNet_dir = asset_dir.joinpath('shapenet')
kitting_dir = asset_dir.joinpath('kitting')
MAX_SHAPENET_SIZE = 0.15
MAX_KITTING_SIZE = 0.10
KITTING_Z_SCALE = 0.3

assets_info = {}

# === Kitting objects ===
for f in os.listdir(kitting_dir):
    if f.endswith('.obj'):
        assert_name = f.split('.')[0]
        # load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(kitting_dir, f))

        # scale the object
        mesh_points = np.asarray(mesh.vertices)
        # get the center of the mesh
        center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2.0
        # shift the origin to center
        mesh.translate(-center)
        # get size along each axis
        size = mesh.get_max_bound() - mesh.get_min_bound()
        scale = MAX_KITTING_SIZE / np.linalg.norm(size)
        # scale the mesh
        dim_scale = np.array([scale, scale, scale * KITTING_Z_SCALE])
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * dim_scale)
        size = size * dim_scale
        o3d.io.write_triangle_mesh(os.path.join(kitting_dir, f'{assert_name}.obj'), mesh)
        # Step: save the size to info
        info = {}
        info['max_size'] = size.tolist()
        info['min_size'] = size.tolist()
        assets_info[assert_name] = info

# == ShapNet objects ===
for f in os.listdir(os.path.join(shapeNet_dir, 'meshes')):
    if f.endswith('.obj'):
        asset_name = f.split('.')[0]
        # load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(shapeNet_dir, 'meshes', f))
        mesh.compute_vertex_normals()

        # Step. 1: rotate along z-axis for 90 degree
        # # rotate along x-axis for 90 degree
        # rot = R.from_euler('x', 90, degrees=True)
        # mesh.rotate(rot.as_matrix(), center=(0, 0, 0))

        # Step. 2: shift the origin to center in z-axis
        mesh_points = np.asarray(mesh.vertices)
        # get the center of the mesh
        center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2.0
        # shift the origin to center
        mesh.translate(-center)
        # get size along each axis
        size = mesh.get_max_bound() - mesh.get_min_bound()
        scale = MAX_SHAPENET_SIZE / np.linalg.norm(size)
        # scale the mesh
        mesh.scale(scale, center=(0, 0, 0))
        size = size * scale
        # resize
        # Step. 3: save the size to info
        info = {}
        info['max_size'] = size.tolist()
        info['min_size'] = size.tolist()
        # add origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # check
        # o3d.visualization.draw_geometries([mesh, origin])
        # save it as "{}_rot.obj".format(f.split('.')[0])
        o3d.io.write_triangle_mesh(os.path.join(shapeNet_dir, 'meshes', f'{asset_name}.obj'), mesh)

        # save info
        assets_info[asset_name] = info

# save info
with open(os.path.join(str(asset_dir), 'assets_info.json'), 'w') as f:
    json.dump(assets_info, f, indent=4)
