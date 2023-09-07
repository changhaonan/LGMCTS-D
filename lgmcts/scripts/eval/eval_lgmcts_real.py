"""Evaluate lgmcts on real robot"""
from __future__ import annotations
import os
import open3d as o3d
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
import lgmcts.utils.misc_utils as utils

def eval_real(real_data_path):
    # load camera
    camera_pose = np.array([[-9.98961852e-01,  4.55540366e-02, -2.20703533e-04,  2.41992141e-02],
                            [4.55544520e-02, 9.98936424e-01, -7.12825762e-03, -4.90078981e-01],
                            [-1.04252110e-04, -7.13091146e-03, -9.99974569e-01, 5.98174172e-01],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]
                           )
    # camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = R.from_euler("xyz", [0, 0, -np.pi * 0.5]).as_matrix() @ camera_pose[:3, :3]

    intrinsics_matrix = np.array([[635.41156006,   0., 644.21557617],
                                  [0.,  634.80944824, 368.45831299],
                                  [0.,    0.,   1.]])
    label = json.load(open(os.path.join(real_data_path, "label.json"), "r"))
    # load images
    depth_scale = 100000.0
    mask = cv2.imread(os.path.join(real_data_path, "mask_image.png"))
    depth = cv2.imread(os.path.join(real_data_path, "depth_image.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16) / depth_scale
    color = cv2.imread(os.path.join(real_data_path, "color_image.png"))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # load pointcloud
    mask_ids = [mask_info["value"] for mask_info in label["mask"] if mask_info["label"] != "background"]
    pcd_list = utils.get_pointcloud_list(color, depth, mask, mask_ids, intrinsics_matrix, np.eye(4, dtype=np.float32))
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    origin.transform(camera_pose)
    # for pcd in pcd_list:
    #     o3d.visualization.draw_geometries([pcd, origin])
    o3d.visualization.draw_geometries(pcd_list + [origin])
    # create instance list

    # init region_sampler
    resolution = 0.002
    pix_padding = 1  # padding for clearance
    bounds = np.array([[0.0, 1.0], [-0.5, 0.5], [0.0, 0.5]])
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    # region_sampler.visualize()
    # color = cv2.imread(os.path.join(real_data_path, "top_down_color.jpg"))
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # depth = cv2.imread(os.path.join(real_data_path, "top_down_depth.jpg"), cv2.IMREAD_UNCHANGED)
    # depth = np.array(depth, dtype=np.uint16) / 10000.0
    # mask = cv2.imread(os.path.join(real_data_path, "top_down_mask.jpg"))
    region_sampler.load_from_pcds(pcd_list, mask_mode="raw_mask", cam2world=np.linalg.inv(camera_pose))
    region_sampler.visualize()
    region_sampler.visualize_3d()

if __name__ == "__main__":
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    real_data_path = os.path.join(root_path, "test_data", "real_000000")
    eval_real(real_data_path)
