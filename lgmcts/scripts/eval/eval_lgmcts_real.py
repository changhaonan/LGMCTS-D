"""Evaluate lgmcts on real robot"""
from __future__ import annotations
import os
import open3d as o3d
import cv2
import numpy as np
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
import lgmcts.utils.real_utils as real_utils


def eval_real(real_data_path):
    # load camera
    camera_pose = np.array([[-9.98961852e-01,  4.55540366e-02, -2.20703533e-04,  2.41992141e-02],
                            [4.55544520e-02, 9.98936424e-01, -7.12825762e-03, -4.90078981e-01],
                            [-1.04252110e-04, -7.13091146e-03, -9.99974569e-01, 5.98174172e-01],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]
                           )

    intrinsics_matrix = np.array([[635.41156006,   0., 644.21557617],
                           [0.,  634.80944824, 368.45831299],
                           [0.,    0.,   1.]])
    # load images
    # mask = cv2.imread(os.path.join(real_data_path, "mask.jpg"))
    # depth = cv2.imread(os.path.join(real_data_path, "depth_image.png"), cv2.IMREAD_ANYDEPTH)
    # color = cv2.imread(os.path.join(real_data_path, "color_image.png"))
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     o3d.geometry.Image(color),
    #     o3d.geometry.Image(depth),
    #     depth_scale=1000.0,  # Depth scale factor (adjust as needed)
    #     depth_trunc=1000.0,  # Depth truncation (adjust as needed)
    #     convert_rgb_to_intensity=False
    # )
    # intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # intrinsics.set_intrinsics(
    #     intrinsics_matrix.shape[1],
    #     intrinsics_matrix.shape[0],
    #     intrinsics_matrix[0, 0],  # fx
    #     intrinsics_matrix[1, 1],  # fy
    #     intrinsics_matrix[0, 2],  # cx
    #     intrinsics_matrix[1, 2]   # cy
    # )
    # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd, intrinsics
    # )
    # o3d.visualization.draw_geometries([pcd])

    color = cv2.imread(os.path.join(real_data_path, "top_down_color.jpg"))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(real_data_path, "top_down_depth.jpg"), cv2.IMREAD_UNCHANGED)
    depth = np.array(depth, dtype=np.uint16) / 10000.0
    mask = cv2.imread(os.path.join(real_data_path, "top_down_mask.jpg"))
    pass

if __name__ == "__main__":
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    real_data_path = os.path.join(root_path, "test_data", "real_000000")
    eval_real(real_data_path)
