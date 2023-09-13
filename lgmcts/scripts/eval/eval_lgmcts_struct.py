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

def dist_p2l(p, o, k):
    """(Vectorized meethod) disance, point to line"""
    op = p - o
    k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
    op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
    op_ver = op - op_proj
    return np.linalg.norm(op_ver, axis=-1)

def eval_real(data_path: str, prompt_path: str, method: str, mask_mode: str, n_samples: int = 10, debug: bool = True, start: int = 0, end: int = 100):
    # Step 1. load the scene
    # camera_pose = np.array([
    #     [-9.99019040e-01,  4.42819236e-02,  2.62008166e-04,  2.40630148e-02],
    #     [4.42787021e-02,  9.98990882e-01, -7.52417562e-03, -4.88996877e-01],
    #     [-5.94928738e-04, -7.50519333e-03, -9.99971659e-01,  5.96053361e-01],
    #     [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
    # ])
    camera_pose = np.array([[-0.99874228,  0.00730198,  0.04960381,  0.428     ],
       [-0.04565842,  0.27632073, -0.95998029,  0.988     ],
       [-0.02071632, -0.96103774, -0.2756398 ,  0.634     ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]
       ])
    
    intrinsics_matrix = np.array([[635.41156006,   0., 644.21557617],
                                  [0.,  634.80944824, 368.45831299],
                                  [0.,    0.,   1.]])
    # label = json.load(open(os.path.join(data_path, "label.json"), "r"))
    # load images
    # depth_scale = 100000.0
    # mask = cv2.imread(os.path.join(data_path, "mask_image.png"))
    # depth = cv2.imread(os.path.join(data_path, "depth_image.png"),
                    #    cv2.IMREAD_UNCHANGED).astype(np.uint16) / depth_scale
    # color = cv2.imread(os.path.join(data_path, "color_image.png"))
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # load pointcloud
    # name_ids = []
    # texture_mapping = {}
    # for mask_info in label["mask"]:
    #     if mask_info["label"] == "background":
    #         continue
    #     name_ids.append((mask_info["label"].split(" ")[0], mask_info["value"]))
    #     if "color" in mask_info:
    #         texture_mapping[mask_info["value"]] = mask_info["color"]
    #     else:
    #         texture_mapping[mask_info["value"]] = "unknown"

    line_dataset_path = "/media/exx/T7 Shield/ICLR23/SFormer-original/StructFormer/data_new_objects_test_split/line-pcd-objs"
    res_line_dataset_path = "/media/exx/T7 Shield/ICLR23/SFormer-original/StructFormer/data_new_objects_test_split/res-line-pcd-objs"
    h5_folders = os.listdir(line_dataset_path)
    natsorted(h5_folders)
    sformer_success_rate = []
    mcts_success_rate = []
    use_sformer_result = False
    mcts_success_result = dict()
    sformer_success_result = dict()
    h5_folders = ['data00751124.h5', 'data00742126.h5']
    for iter in tqdm.tqdm(range(len(h5_folders[start:end]))):
        h5_folder = h5_folders[start:end][iter]
        print("h5 file:", h5_folder)
        pcd_list = []
        obj_pc_centers = []
        with open(f"{line_dataset_path}/{h5_folder}/obj_pcd_list.pkl", "rb") as f:
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
                # obj_pc_centers.append(torch.mean(xyz, dim=0).numpy())
                pcd_list.append(obj_pcd)

        name_ids = None 
        goals = []
        goal_pose_sformer = None
        curr_pose_sformer = None
        sformer_action_list = []
        obj_poses_pattern = []
        
        # data00717669 
        # data00700001.h5
        with open(f"{line_dataset_path}/{h5_folder}/name_ids.pkl", "rb") as f:
            name_ids = pickle.load(f)
        texture_mapping = None
        with open(f"{line_dataset_path}/{h5_folder}/texture_mapping.pkl", "rb") as f:
            texture_mapping = pickle.load(f)
        with open(f"{line_dataset_path}/{h5_folder}/goal.pkl", "rb") as f:
            goals.append(pickle.load(f))
        with open(f"{line_dataset_path}/{h5_folder}/goal_pose_0.pkl", "rb") as f:
            goal_pose_sformer = pickle.load(f)
        with open(f"{line_dataset_path}/{h5_folder}/current_pose_0.pkl", "rb") as f:
            curr_pose_sformer = pickle.load(f)

        if use_sformer_result:
            for index, id in enumerate(goals[0]["obj_ids"]):
                sformer_action_list.append({"obj_id": id, "old_pose": curr_pose_sformer[index], "new_pose": goal_pose_sformer[index]})
        
        # pcd_list = utils.get_pointcloud_list(color, depth, mask, name_ids,
        #                                      intrinsics_matrix, np.eye(4, dtype=np.float32))
        
        # init region_sampler
        resolution = 0.002
        pix_padding = 1  # padding for clearance
        # bounds = np.array([[-0.4, 0.4], [-0.5, 0.5], [0.0, 0.5]])  # (height, width, depth)
        bounds = np.array([[-0.8, 0.8], [-1.0, 1.0], [-0.5, 1.0]])  # (height, width, depth)
        region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
        region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="raw_mask")
        if debug:
            region_sampler.visualize()
            region_sampler.visualize_3d(show_origin=True, obj_center=obj_pc_center)
        init_objects_poses = region_sampler.get_object_poses()
        obj_id_reverse_mapping = {}
        for name_id in name_ids:
            obj_id_reverse_mapping[name_id[1]] = {"obj_name": name_id[0], "texture_name": texture_mapping[name_id[0]]}
        # Step 2. parse the goal using LLM
        # FIXME: manually set the goal for now
        use_llm = True
        run_llm = True
        # encode_ids_to_llm = True
        # # Generate goals using llm and object selector
        # prompt_goals = gen_prompt_goal_from_llm(prompt_path, use_llm=use_llm,
        #                                         run_llm=run_llm, encode_ids_to_llm=encode_ids_to_llm, obj_id_reverse_mappings=[obj_id_reverse_mapping], debug=debug)

        # goals = prompt_goals[0]
        # goals = [
        #     # {"type": "pattern:rectangle", "obj_ids": [3, 4, 5, 6]},
        #     {"type": "pattern:line", "obj_ids": [4, 1, 2], "anchor_id" : },
        # ]
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
        if not use_sformer_result:
            action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug, max_iter=20000, seed=1)
        else:
            action_list = sformer_action_list # Checking SFORMER action list
        for entry in action_list:
            if use_sformer_result:
                obj_poses_pattern.append(matrix_to_xyz_quaternion(entry["new_pose"]))
            else:
                obj_poses_pattern.append(entry["new_pose"])
        # print("Plan finished!")
        region_sampler.set_object_poses(init_objects_poses)
        if debug:
            region_sampler.visualize()
        export_action_list = []
        for step in action_list:
            if use_sformer_result:
                region_sampler.set_object_pose(step["obj_id"], matrix_to_xyz_quaternion(step["new_pose"]))
                if debug:
                    region_sampler.visualize()
            else:
                region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
                if debug:
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
        if not use_sformer_result:
            with open(os.path.join(data_path, "action_list.json"), "w") as f:
                json.dump(export_action_list, f)

        # Step 4. Calculate Success Rate

        obj_poses_pattern = np.vstack(obj_poses_pattern)
        # get the up most and low most points first"""
        lo_idx = np.argmax(obj_poses_pattern[:, 1], axis=-1)
        hi_idx = np.argmin(obj_poses_pattern[:, 1], axis=-1)
        lo_pose = obj_poses_pattern[lo_idx, :2]
        hi_pose = obj_poses_pattern[hi_idx, :2]
        k = (hi_pose - lo_pose) / np.linalg.norm(hi_pose - lo_pose)
        o = hi_pose
        threshold = 0.1 
        dists = dist_p2l(obj_poses_pattern[:, :2], o[None, :], k[None, :])
        # pattern_dists.append(np.max(dists))
        status = not (np.max(dists) > threshold)
        
        if status:
            if use_sformer_result:
                sformer_success_result[h5_folder] = 1
                sformer_success_rate.append(1)
            else:
                mcts_success_result[h5_folder] = 1
                mcts_success_rate.append(1)
            print("Line pattern check passed!", "Max Dist:", np.max(dists))
        else:
            if use_sformer_result:
                sformer_success_result[h5_folder] = 0
                sformer_success_rate.append(0)
            else:
                mcts_success_result[h5_folder] = 0
                mcts_success_rate.append(0)
            print("Line pattern check failed!", "Max Dist:", np.max(dists))
        # if not status:
        #     print("Line pattern check failed!", "Max Dist:", np.max(dists))
        #     pass
        # else:
    if not use_sformer_result:    
        mcts_success_rate = np.array(mcts_success_rate)
        mcts_success_result["success_rate"] = np.mean(mcts_success_rate)*100
        with open(f"{res_line_dataset_path}/mcts_success_result_{start}_to_{end}.pkl", "wb") as f:
            pickle.dump(mcts_success_result, f)
        print("MCTS Success Rate:", np.mean(mcts_success_rate)*100)
    else:
        sformer_success_rate = np.array(sformer_success_rate)
        sformer_success_result["success_rate"] = np.mean(sformer_success_rate)*100
        with open(f"{res_line_dataset_path}/sformer_success_result_{start}_to_{end}.pkl", "wb") as f:
            pickle.dump(sformer_success_result, f)
        print("SFormer Success Rate:", np.mean(sformer_success_rate)*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=2, help="End index")
    args = parser.parse_args()

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    real_data_path = os.path.join(root_path, "test_data", "real_000000")
    prompt_path = f"{root_path}/output/struct_rearrange"
    eval_real(real_data_path, prompt_path, args.method, args.mask_mode, args.n_samples, args.debug, args.start, args.end)
