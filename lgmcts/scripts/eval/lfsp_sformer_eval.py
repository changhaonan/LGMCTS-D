"""Evaluate LFSP on Structformer dataset"""
from __future__ import annotations
import os
import open3d as o3d
import numpy as np
import argparse
import pickle
import torch
import copy
import random
import tqdm
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.components.semantic_patterns import REMAPPING_PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS
from lgmcts.components.llm_chatgpt import ChatGPTAPI

def eval(data_path: str, pattern_name: str = "line", start: int=0, end: int = 25):
    h5_folders = os.listdir(data_path)
    h5_full = os.listdir(data_path)
    h5_folders = []
    for h5_en in h5_full:
        if not h5_en.endswith(".pkl") and h5_en.startswith("data00"):
            h5_folders.append(h5_en)
    natsorted(h5_folders)
        
    # ceiling 10% of 4295 line scenes 
    # ceiling 10% of 3416 circle scenes
    # ceiling 10% of 1335 tower scenes
    # ceiling 10% of 2440 dinner scenes
    num_scenes = {
        "line" : 430,  
        "circle" : 342,
        "tower" : 134,
        "dinner" : 244
    }
    random.seed(45)
    # 4 used for example prompting the LLM
    h5_folders = random.sample(h5_folders[4:], num_scenes[pattern_name])
    mcts_success_rate = []
    if end == -1:
        save_name = "llm_{pattern_name}_res_full.pkl"
    else:
        save_name = f"llm_{pattern_name}_res_{start}_to_{end}.pkl"
    with open(f"{data_path}/{save_name}","rb") as fp:
        llm_result = pickle.load(fp)
    h5_folders = h5_folders[start:end]
    except_block = 0
    for iter in tqdm.tqdm(range(len(llm_result))):
        if llm_result[iter] is None:
            missed += 1
            print("========>>> Missed:", missed)
            continue
        h5_folder = llm_result[iter][0]["scene_id"]
        pcd_list = []
        obj_pc_centers = []
        with open(f"{data_path}/{h5_folder}/obj_pcd_list.pkl", "rb") as f:
            pcd_info = pickle.load(f)
            for xyz, color in zip(pcd_info[0], pcd_info[1]):
                # Calculate the center of obj_pc_centers
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

        with open(f"{data_path}/{h5_folder}/name_ids.pkl", "rb") as f:
            name_ids = pickle.load(f)
        with open(f"{data_path}/{h5_folder}/goal.pkl", "rb") as f:
            goals.append(pickle.load(f))
 
        mod_pcd_list = []
        sort_name_ids = copy.deepcopy(name_ids)
        sort_name_ids.sort(key=lambda x: x[1])
        for obj_id in goals[0]["obj_ids"]:
            for index, name_id in enumerate(sort_name_ids):
                if name_id[1] == obj_id:
                    mod_pcd_list.append(pcd_list[index])
            
        pcd_list = mod_pcd_list

        # check semantic pattern
        new_goals = []
        for goal in goals:
            pattern = pattern_name  # set it from the argument
            if pattern == "dinner":
                pattern = "dinner_v2"
            if pattern in REMAPPING_PATTERN_DICT:
                new_goal = REMAPPING_PATTERN_DICT[pattern].parse_goal(name_ids=name_ids)
                new_goals += new_goal
            else:
                new_goals.append(goal)
        goals = new_goals
        # init region_sampler
        resolution = 0.01
        pix_padding = 1  # padding for clearance
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.0]])  # (height, width, depth)
        region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
        region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="convex_hull")

        init_pose = region_sampler.get_object_poses() 
        region_sampler.set_object_poses(init_pose)
        action_list = llm_result[iter]
        status = None
        for step in action_list:
            region_sampler.set_object_pose(step["obj_id"], step["new_pose"])
        overall_status = 0
        for goal in goals:
            obj_poses = {}
            for entry in action_list:
                if entry["obj_id"] in goal["obj_ids"]:
                    obj_poses[entry['obj_id']] = (entry["new_pose"])
            pattern_info = {"threshold": 0.05}
            pattern_info["obj_ids"] = goal["obj_ids"]
            if "spatial_label" in goal:
                pattern_info["spatial_label"] = goal["spatial_label"]
            try:

                pattern_status = PATTERN_DICT[goal["type"].split(
                    ":")[-1]].check(obj_poses=obj_poses, pattern_info=pattern_info)
                not_collision = not region_sampler.check_collision(goal["obj_ids"])
                if goal["type"] == "pattern:tower":
                    not_collision = True
                status = pattern_status and not_collision
                if status:
                    overall_status += 1
            except:
                except_block += 1
                print(f"Into the except block {except_block} times....")
        print("OVerall Status:", overall_status, "len of goals:", len(goals))
        if overall_status == len(goals):
            mcts_success_rate.append(1)
        else:
            mcts_success_rate.append(0)       

    mcts_success_rate = np.array(mcts_success_rate)
    assert len(mcts_success_rate) == len(llm_result)
    result = {
        "start" : start,
        "end" : end,
        "num" : len(llm_result),
        "success_rate": np.mean(mcts_success_rate)*100
    }
    print(result)
    if end == -1:
        save_name = "llm_{pattern_name}_score_full.pkl"
    else:
        save_name = f"llm_{pattern_name}_score_{start}_to_{end}.pkl"
    with open(f"{data_path}/{save_name}","wb") as fp:
        pickle.dump(result, fp)
    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--res_path", type=str, default=None, help="Path to the prompt")
    parser.add_argument("--pattern", type=str, default="line", help="Pattern")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=5, help="End index")
    args = parser.parse_args()

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args.data_path = os.path.join(root_path, f"output/lfsp/eval_single_pattern/{args.pattern}-pcd-objs")
    args.res_path = os.path.join(root_path, f"output/lfsp/eval_single_pattern/res-{args.pattern}-pcd-objs")
    eval(args.data_path, pattern_name=args.pattern, start=args.start, end=args.end)