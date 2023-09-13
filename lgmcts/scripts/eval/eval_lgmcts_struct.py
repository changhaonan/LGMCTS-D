"""Evaluate lgmcts on StructTransformer"""
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
from lgmcts.scripts.data_generation.llm_parse import gen_prompt_goal_from_llm
import lgmcts.utils.misc_utils as utils


def eval_real(data_path: str, prompt_path: str, method: str, mask_mode: str, n_samples: int = 10, debug: bool = True):
    # Step 1. load the scene
    camera_pose = np.array([
        [-9.99019040e-01,  4.42819236e-02,  2.62008166e-04,  2.40630148e-02],
        [4.42787021e-02,  9.98990882e-01, -7.52417562e-03, -4.88996877e-01],
        [-5.94928738e-04, -7.50519333e-03, -9.99971659e-01,  5.96053361e-01],
        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    ])
    intrinsics_matrix = np.array([[635.41156006,   0., 644.21557617],
                                  [0.,  634.80944824, 368.45831299],
                                  [0.,    0.,   1.]])
    label = json.load(open(os.path.join(data_path, "label.json"), "r"))
    # load images
    depth_scale = 100000.0
    mask = cv2.imread(os.path.join(data_path, "mask_image.png"))
    depth = cv2.imread(os.path.join(data_path, "depth_image.png"),
                       cv2.IMREAD_UNCHANGED).astype(np.uint16) / depth_scale
    color = cv2.imread(os.path.join(data_path, "color_image.png"))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # load pointcloud
    name_ids = []
    texture_mapping = {}
    for mask_info in label["mask"]:
        if mask_info["label"] == "background":
            continue
        name_ids.append((mask_info["label"].split(" ")[0], mask_info["value"]))
        if "color" in mask_info:
            texture_mapping[mask_info["value"]] = mask_info["color"]
        else:
            texture_mapping[mask_info["value"]] = "unknown"
    pcd_list = utils.get_pointcloud_list(color, depth, mask, name_ids,
                                         intrinsics_matrix, np.eye(4, dtype=np.float32))
    # init region_sampler
    resolution = 0.002
    pix_padding = 1  # padding for clearance
    bounds = np.array([[-0.4, 0.4], [-0.5, 0.5], [0.0, 0.5]])  # (height, width, depth)
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="raw_mask")
    region_sampler.visualize()
    init_objects_poses = region_sampler.get_object_poses()
    obj_id_reverse_mapping = {}
    for name_id in name_ids:
        obj_id_reverse_mapping[name_id[1]] = {"obj_name": name_id[0], "texture_name": texture_mapping[name_id[1]]}
    # Step 2. parse the goal using LLM
    # FIXME: manually set the goal for now
    use_llm = True
    run_llm = True
    # encode_ids_to_llm = True
    # # Generate goals using llm and object selector
    # prompt_goals = gen_prompt_goal_from_llm(prompt_path, use_llm=use_llm,
    #                                         run_llm=run_llm, encode_ids_to_llm=encode_ids_to_llm, obj_id_reverse_mappings=[obj_id_reverse_mapping], debug=debug)

    # goals = prompt_goals[0]
    goals = [
        {"type": "pattern:rectangle", "obj_ids": [3, 4, 5, 6]},
        {"type": "pattern:line", "obj_ids": [4, 1, 2]},
    ]
    sampled_ids = []
    L = []
    for goal in goals:
        goal_obj_ids = goal["obj_ids"]
        goal_pattern = goal["type"].split(":")[-1]
        print(f"Goal: {goal_pattern}; {goal_obj_ids}")

        ordered = False
        for _i, goal_obj_id in enumerate(goal_obj_ids):
            sample_info = {"ordered": ordered}
            if goal["type"] == "pattern:spatial":
                sample_info["spatial_label"] = goal["spatial_label"]
            if goal_obj_id in sampled_ids:
                # meaning that this object has been sampled before
                ordered = True
                continue
            sample_data = SampleData(goal_pattern, goal_obj_id, goal["obj_ids"], {}, sample_info)
            L.append(sample_data)
            sampled_ids.append(goal_obj_id)

    # Step 3. generate & exectue plan
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)
    action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug, max_iter=20000, seed=1)
    print("Plan finished!")
    region_sampler.set_object_poses(init_objects_poses)
    region_sampler.visualize()
    export_action_list = []
    for step in action_list:
        region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
        region_sampler.visualize()
        #
        pose0_position = camera_pose[:3, :3] @ step["old_pose"][:3] + camera_pose[:3, 3]
        pose0_position[2] = 0.0
        pose1_position = camera_pose[:3, :3] @ step["new_pose"][:3] + camera_pose[:3, 3]
        pose1_position[2] = 0.05
        action = {
            "obj_id": int(step["obj_id"]),
            "pose0_position": pose0_position.tolist(),
            "pose0_rotation": step["old_pose"][3:].tolist(),
            "pose1_position": pose1_position.tolist(),
            "pose1_rotation": step["new_pose"][3:].tolist(),
        }
        export_action_list.append(action)
    # export to json
    with open(os.path.join(data_path, "action_list.json"), "w") as f:
        json.dump(export_action_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    real_data_path = os.path.join(root_path, "test_data", "real_000005", "output")
    prompt_path = f"{root_path}/output/struct_rearrange"
    eval_real(real_data_path, prompt_path, args.method, args.mask_mode, args.n_samples, args.debug)
