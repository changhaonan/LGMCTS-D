"""Evaluate lgmcts on real robot"""
from __future__ import annotations
import os
import open3d as o3d
import cv2
import numpy as np
import json
import argparse
from scipy.spatial.transform import Rotation as R
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
import lgmcts.utils.misc_utils as utils


def eval_real(real_data_path: str, method: str, mask_mode: str, n_samples: int = 10, debug: bool = True):
    # Step 1. load the scene
    camera_pose = np.array([[-9.98961852e-01,  4.55540366e-02, -2.20703533e-04,  2.41992141e-02],
                            [4.55544520e-02, 9.98936424e-01, -7.12825762e-03, -4.90078981e-01],
                            [-1.04252110e-04, -7.13091146e-03, -9.99974569e-01, 5.98174172e-01],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]
                           )
    intrinsics_matrix = np.array([[635.41156006,   0., 644.21557617],
                                  [0.,  634.80944824, 368.45831299],
                                  [0.,    0.,   1.]])
    label = json.load(open(os.path.join(real_data_path, "label.json"), "r"))
    # load images
    depth_scale = 100000.0
    mask = cv2.imread(os.path.join(real_data_path, "mask_image.png"))
    depth = cv2.imread(os.path.join(real_data_path, "depth_image.png"),
                       cv2.IMREAD_UNCHANGED).astype(np.uint16) / depth_scale
    color = cv2.imread(os.path.join(real_data_path, "color_image.png"))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # load pointcloud
    name_ids = [(mask_info["label"], mask_info["value"])
                for mask_info in label["mask"] if mask_info["label"] != "background"]
    pcd_list = utils.get_pointcloud_list(color, depth, mask, name_ids,
                                         intrinsics_matrix, np.eye(4, dtype=np.float32))
    # init region_sampler
    resolution = 0.002
    pix_padding = 1  # padding for clearance
    bounds = np.array([[-0.35, 0.35], [-0.5, 0.5], [0.0, 0.5]])  # (height, width, depth)
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="raw_mask")
    init_objects_poses = region_sampler.get_object_poses()
    # Step 2. load the goal
    # FIXME: manually set the goal for now
    goals = []
    goal = {"type": "pattern:line", "obj_ids": [1, 2, 3, 4]}
    goals.append(goal)

    L = []
    for goal in goals:
        goal_obj_ids = goal["obj_ids"]
        goal_pattern = goal["type"].split(":")[-1]
        print(f"Goal: {goal_pattern}; {goal_obj_ids}")

        for _i, goal_obj_id in enumerate(goal_obj_ids):
            sample_info = {}
            if goal_pattern == "spatial":
                # spatial only sample the second obj
                if _i == 0:
                    continue
                else:
                    sample_info = {"spatial_label": goal["spatial_label"], "ordered": True}
            sample_data = SampleData(goal_pattern, goal_obj_id, goal["obj_ids"], {}, sample_info)
            L.append(sample_data)

    # Step 3. generate & exectue plan
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)
    action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug)
    region_sampler.set_object_poses(init_objects_poses)
    region_sampler.visualize()
    for step in action_list:
        region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
        region_sampler.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    real_data_path = os.path.join(root_path, "test_data", "real_000000")
    eval_real(real_data_path, args.method, args.mask_mode, args.n_samples, args.debug)
