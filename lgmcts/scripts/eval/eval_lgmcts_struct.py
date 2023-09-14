"""Evaluate lgmcts on StructTransformer"""
from __future__ import annotations
import os
import open3d as o3d
import cv2
import numpy as np
import json
import argparse
import pickle
import torch
import copy
import pathlib
import tqdm
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.components.semantic_patterns import SEMANTIC_PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
from lgmcts.scripts.data_generation.llm_parse import gen_prompt_goal_from_llm
import lgmcts.utils.misc_utils as utils


def matrix_to_xyz_quaternion(matrix):
    # Extract translation
    x, y, z = matrix[0:3, 3]

    # Extract 3x3 rotation matrix
    rotation_matrix = matrix[0:3, 0:3]

    # Compute quaternion from rotation matrix
    q_w = np.sqrt(np.trace(rotation_matrix) + 1) / 2.0
    q_x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * q_w)
    q_y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * q_w)
    q_z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * q_w)

    return np.array([x, y, z, q_x, q_y, q_z, q_w])


def extract_euler_angles(M):

    # Assuming M is a 4x4 transformation matrix
    x, y, z = M[0:3, 3]
    # Extract the 3x3 rotation matrix
    R = M[:3, :3]

    # Extract individual elements of the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    # Calculate yaw, pitch, and roll
    yaw = np.arctan2(r21, r11)
    pitch = np.arctan2(-r31, np.sqrt(r32 ** 2 + r33 ** 2))
    roll = np.arctan2(r32, r33)

    # Convert from radians to degrees
    yaw, pitch, roll = np.degrees([yaw, pitch, roll])

    return np.array([x, y, z, yaw, pitch, roll])


def dist_p2l(p, o, k):
    """(Vectorized meethod) disance, point to line"""
    op = p - o
    k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
    op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
    op_ver = op - op_proj
    return np.linalg.norm(op_ver, axis=-1)


def eval(data_path: str, res_path: str, method: str, mask_mode: str, n_samples: int = 10, debug: bool = True, start: int = 0, end: int = 100):
    # Step 1. load the scene
    camera_pose = np.array([[-0.99874228,  0.00730198,  0.04960381,  0.428],
                            [-0.04565842,  0.27632073, -0.95998029,  0.988],
                            [-0.02071632, -0.96103774, -0.2756398,  0.634],
                            [0.,  0.,  0.,  1.]
                            ])

    h5_folders = os.listdir(data_path)
    natsorted(h5_folders)
    sformer_success_rate = []
    mcts_success_rate = []
    if method == "sformer":
        use_sformer_result = True
    else:
        use_sformer_result = False
    start = 0
    end = len(h5_folders)
    mcts_success_result = dict()
    sformer_success_result = dict()
    h5_folders = ['data00702857.h5']
    failures = []
    for iter in tqdm.tqdm(range(len(h5_folders[start:end]))):
        h5_folder = h5_folders[start:end][iter]
        print("h5 file:", h5_folder)
        pcd_list = []
        obj_pc_centers = []
        with open(f"{data_path}/{h5_folder}/obj_pcd_list.pkl", "rb") as f:
            pcd_info = pickle.load(f)
            for xyz, color in zip(pcd_info[0], pcd_info[1]):
                # Calcualte the center of obj_pc_centers
                obj_pc_centers.append(torch.mean(xyz, dim=0).numpy())
            obj_pc_centers = np.array(obj_pc_centers)
            obj_pc_center = np.mean(obj_pc_centers, axis=0)

            for xyz, color in zip(pcd_info[0], pcd_info[1]):
                obj_pcd = o3d.geometry.PointCloud()
                obj_pcd.points = o3d.utility.Vector3dVector(xyz-obj_pc_center)
                obj_pcd.colors = o3d.utility.Vector3dVector(color)
                pcd_list.append(obj_pcd)
        name_ids = None
        goals = []
        goal_pose_sformer = None
        curr_pose_sformer = None
        sformer_action_list = []
        obj_poses_pattern = []

        with open(f"{data_path}/{h5_folder}/name_ids.pkl", "rb") as f:
            name_ids = pickle.load(f)
        texture_mapping = None
        with open(f"{data_path}/{h5_folder}/texture_mapping.pkl", "rb") as f:
            texture_mapping = pickle.load(f)
        with open(f"{data_path}/{h5_folder}/goal.pkl", "rb") as f:
            goals.append(pickle.load(f))
        if "-diffusion" in data_path:
            with open(f"{data_path}/{h5_folder}/goal_pose.pkl", "rb") as f:
                goal_pose_sformer = pickle.load(f)
                assert len(goal_pose_sformer) == 8
                goal_pose_sformer = goal_pose_sformer[1:]  # First pose is the pose of the structure frame
        else:
            with open(f"{data_path}/{h5_folder}/goal_pose_0.pkl", "rb") as f:
                goal_pose_sformer = pickle.load(f)

        if use_sformer_result:
            with open(f"{data_path}/{h5_folder}/goal_pose_0.pkl", "rb") as f:
                goal_pose_sformer = pickle.load(f)
            with open(f"{data_path}/{h5_folder}/current_pose_0.pkl", "rb") as f:
                curr_pose_sformer = pickle.load(f)
            for index, id in enumerate(goals[0]["obj_ids"]):
                sformer_action_list.append({"obj_id": id, "new_pose": goal_pose_sformer[index]})
        else:
            # check semantic pattern
            new_goals = []
            for goal in goals:
                pattern = goal["type"].split(":")[-1]
                if pattern in SEMANTIC_PATTERN_DICT:
                    new_goal = SEMANTIC_PATTERN_DICT[pattern].parse_goal(name_ids)
                    new_goals += new_goal
                else:
                    new_goals.append(goal)
            goals = new_goals
        # init region_sampler
        resolution = 0.002 if use_sformer_result else 0.01
        pix_padding = 1  # padding for clearance
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.0]])  # (height, width, depth)
        region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
        region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="convex_hull")
        if debug:
            region_sampler.visualize()
        init_objects_poses = region_sampler.get_object_poses()
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
        print(region_sampler.check_collision())
        region_sampler.visualize()
        region_sampler.visualize_3d()
        # Step 3. generate & exectue plan
        check_goal_idx = 0
        sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)
        if not use_sformer_result:
            action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug, max_iter=10000, seed=1, is_virtual=False)
        else:
            action_list = sformer_action_list  # Checking SFORMER action list
        for entry in action_list:
            if use_sformer_result:
                obj_poses_pattern.append(extract_euler_angles(entry["new_pose"]))
            else:
                if entry["obj_id"] in goals[check_goal_idx]["obj_ids"]:
                    obj_poses_pattern.append(entry["new_pose"])
        region_sampler.set_object_poses(init_objects_poses)
        if debug:
            region_sampler.visualize()

        result_pcd_list = []
        new_name_ids = []
        for step in action_list:
            if use_sformer_result:
                pcd = region_sampler.get_object_pcd(step["obj_id"])
                pcd = copy.deepcopy(pcd)
                R = step["new_pose"][:3, :3]
                t = step["new_pose"][:3, 3]
                pcd.rotate(R, center=pcd.get_center())
                pcd.translate(t)
                result_pcd_list.append(pcd)
                for name_id in name_ids:
                    if name_id[1] == step["obj_id"]:
                        new_name_ids.append([name_id[0], step["obj_id"]])
                        break
            else:
                region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
                if debug:
                    region_sampler.visualize()
        utils.plot_pcd_o3d(result_pcd_list, show_origin=True)
        if use_sformer_result:
            region_sampler.reset()
            region_sampler.load_from_pcds(result_pcd_list, name_ids=new_name_ids, mask_mode="convex_hull")
            if debug:
                region_sampler.visualize()
                region_sampler.visualize_3d()

        # Step 4. Calculate Success Rate
        obj_poses_pattern = np.vstack(obj_poses_pattern)
        pattern_info = {"threshold": 0.05}
        pattern_status = PATTERN_DICT[goals[check_goal_idx]["type"].split(
            ":")[-1]].check(obj_poses_pattern=obj_poses_pattern, pattern_info=pattern_info)
        not_collision = not region_sampler.check_collision(goals[check_goal_idx]["obj_ids"])
        if goals[0]["type"] == "pattern:tower":
            not_collision = True
        status = pattern_status and not_collision
        if status:
            if use_sformer_result:
                sformer_success_result[h5_folder] = {"success_rate": 1, "pattern_status": pattern_status, "not_collision": not_collision}
                sformer_success_rate.append(1)
            else:
                mcts_success_result[h5_folder] = {"success_rate": 1, "pattern_status": pattern_status, "not_collision": not_collision}
                mcts_success_rate.append(1)
        else:
            if use_sformer_result:
                sformer_success_result[h5_folder] = {"success_rate": 0, "pattern_status": pattern_status, "not_collision": not_collision}
                sformer_success_rate.append(0)
                failures.append({"File": h5_folder, "Pattern Status": pattern_status, "Collision Status": not not_collision})
            else:
                mcts_success_result[h5_folder] = {"success_rate": 0, "pattern_status": pattern_status, "not_collision": not_collision}
                mcts_success_rate.append(0)
                failures.append({"File": h5_folder, "Pattern Status": pattern_status, "Collision Status": not not_collision})

    for fail_case in failures:
        print(fail_case)
    if not use_sformer_result:
        mcts_success_rate = np.array(mcts_success_rate)
        mcts_success_result["success_rate"] = np.mean(mcts_success_rate)*100
        with open(f"{res_path}/mcts_success_result_{start}_to_{end}.pkl", "wb") as f:
            pickle.dump(mcts_success_result, f)
        print("MCTS Success Rate:", np.mean(mcts_success_rate)*100)
    else:
        sformer_success_rate = np.array(sformer_success_rate)
        sformer_success_result["success_rate"] = np.mean(sformer_success_rate)*100
        with open(f"{res_path}/sformer_success_result_{start}_to_{end}.pkl", "wb") as f:
            pickle.dump(sformer_success_result, f)
        print("SFormer Success Rate:", np.mean(sformer_success_rate)*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--res_path", type=str, default=None, help="Path to the prompt")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=2, help="End index")
    args = parser.parse_args()

    debug = True
    args.method = "sformer"
    pattern = "line"
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    args.data_path = os.path.join(root_path, f"output/eval_single_pattern/{pattern}-pcd-objs")
    args.res_path = os.path.join(root_path, f"output/eval_single_pattern/res-{pattern}-pcd-objs")
    eval(args.data_path, args.res_path, args.method, args.mask_mode, args.n_samples, debug, args.start, args.end)
