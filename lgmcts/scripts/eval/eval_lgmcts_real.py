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
from lgmcts.scripts.data_generation.llm_parse import gen_prompt_goal_from_llm
import lgmcts.utils.misc_utils as utils


def eval_real(data_path: str, prompt_path: str, method: str, mask_mode: str, n_samples: int = 10, debug: bool = True):
    # Step 1. load the scene
    camera_pose = np.array([[-9.98961852e-01,  4.55540366e-02, -2.20703533e-04,  2.41992141e-02],
                            [4.55544520e-02, 9.98936424e-01, -7.12825762e-03, -4.90078981e-01],
                            [-1.04252110e-04, -7.13091146e-03, -9.99974569e-01, 5.98174172e-01],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],]
                           )
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
    name_ids = [(mask_info["label"], mask_info["value"])
                for mask_info in label["mask"] if mask_info["label"] != "background"]
    pcd_list = utils.get_pointcloud_list(color, depth, mask, name_ids,
                                         intrinsics_matrix, np.eye(4, dtype=np.float32))
    # init region_sampler
    resolution = 0.002
    pix_padding = 1  # padding for clearance
    bounds = np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 0.5]])  # (height, width, depth)
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="raw_mask")
    region_sampler.visualize()
    init_objects_poses = region_sampler.get_object_poses()
    # create an object id reverse mapping
    texture_mapping = {
        "toothpaste": "ruby blue",
        "smartphone2": "pearl white",
        "cube": "yellow",
        "ketchup bottle": "red",
        "bottle": "yellow",
        "ranch bottle": "white green blend",
        "dessert box2": "chocolate",
        "smartphone1": "graphite black",
        "dessert box1": "strawberry splash",
        "box":  "yellow"
    }
    obj_id_reverse_mapping = {}
    for name_id in name_ids:
        obj_id_reverse_mapping[name_id[1]] = {"obj_name": name_id[0], "texture_name": texture_mapping[name_id[0]]}
    # Step 2. parse the goal using LLM
    # FIXME: manually set the goal for now
    use_llm = True
    run_llm = True
    encode_ids_to_llm = True
    # Generate goals using llm and object selector
    prompt_goals = gen_prompt_goal_from_llm(prompt_path, use_llm=use_llm,
                                            run_llm=run_llm, encode_ids_to_llm=encode_ids_to_llm, obj_id_reverse_mappings=[obj_id_reverse_mapping], debug=debug)

    goals = prompt_goals[0]
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
    action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug)
    region_sampler.set_object_poses(init_objects_poses)
    region_sampler.visualize()
    for step in action_list:
        region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
        region_sampler.visualize()


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
    real_data_path = os.path.join(root_path, "test_data", "real_000000")
    prompt_path = f"{root_path}/output/struct_rearrange"
    eval_real(real_data_path, prompt_path, args.method, args.mask_mode, args.n_samples, args.debug)
