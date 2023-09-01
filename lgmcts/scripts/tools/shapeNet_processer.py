"""Process shapeNet data"""
import importlib_resources
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

asset_dir = importlib_resources.files('lgmcts.assets').joinpath('shapenet')
for f in os.listdir(os.path.join(asset_dir, 'meshes')):
    if f.endswith('.obj'):
        # load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(asset_dir, 'meshes', f))
        mesh.compute_vertex_normals()

        # rotate along x-axis for 90 degree
        rot = R.from_euler('x', 90, degrees=True)
        mesh.rotate(rot.as_matrix(), center=(0, 0, 0))

        # add origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        # check
        # o3d.visualization.draw_geometries([mesh, origin])
        # save it as "{}_rot.obj".format(f.split('.')[0])
        o3d.io.write_triangle_mesh(os.path.join(asset_dir, 'meshes', '{}.obj'.format(f.split('.')[0])), mesh)


