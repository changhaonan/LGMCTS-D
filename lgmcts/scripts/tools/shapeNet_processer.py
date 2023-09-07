"""Process shapeNet data"""
import importlib_resources
import os
import numpy as np
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R

asset_dir = importlib_resources.files('lgmcts.assets').joinpath('shapenet')
max_size = 0.15
for f in os.listdir(os.path.join(asset_dir, 'meshes')):
    if f.endswith('.obj'):
        # load info json
        json_file = os.path.join(asset_dir, 'meshes', '{}.json'.format(f.split('.')[0]))
        info = json.load(open(json_file, 'r'))
        # load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(asset_dir, 'meshes', f))
        mesh.compute_vertex_normals()

        ## Step. 1: rotate along z-axis for 90 degree
        # # rotate along x-axis for 90 degree
        # rot = R.from_euler('x', 90, degrees=True)
        # mesh.rotate(rot.as_matrix(), center=(0, 0, 0))

        ## Step. 2: shift the origin to center in z-axis
        mesh_points = np.asarray(mesh.vertices)
        # get the center of the mesh
        center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2.0
        # shift the origin to center
        mesh.translate(-center)
        # get size along each axis
        size = mesh.get_max_bound() - mesh.get_min_bound()
        scale = max_size / np.linalg.norm(size)
        # scale the mesh
        mesh.scale(scale, center=(0, 0, 0))
        size = size * scale
        # resize
        ## Step. 3: save the size to info
        info['max_size'] = size.tolist()
        info['min_size'] = size.tolist()
        # save info
        json.dump(info, open(json_file, 'w'))
        # add origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # check
        # o3d.visualization.draw_geometries([mesh, origin])
        # save it as "{}_rot.obj".format(f.split('.')[0])
        o3d.io.write_triangle_mesh(os.path.join(asset_dir, 'meshes', '{}.obj'.format(f.split('.')[0])), mesh)


