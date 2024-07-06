"""Generate Prompt Database for LFSP on Structformer scenes"""
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
from lgmcts.components.semantic_patterns import REMAPPING_PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
from lgmcts.scripts.data_generation.llm_parse import gen_prompt_goal_from_llm
import lgmcts.utils.misc_utils as utils


def eval(data_path: str, res_path: str, n_samples: int = 10, start: int = 0, end: int = 100, pattern_name: str = "line"):
    # Step 1. load the scene
    h5_folders = os.listdir(data_path)
    h5_full = os.listdir(data_path)
    h5_folders = []
    for h5_en in h5_full:
        if not h5_en.endswith(".pkl") and h5_en.startswith("data00"):
            h5_folders.append(h5_en)
    natsorted(h5_folders)
    mcts_success_rate = []
    start = 0
    end = len(h5_folders)
    mcts_success_result = dict()
    failures = []
    fp = open(f"{data_path}/prompt_example_direct.txt", "w")
    prompt_full = "<root>\n"
    counter = 0
    for iter in tqdm.tqdm(range(len(h5_folders[start:end]))):
        h5_folder = h5_folders[start:end][iter]
        print("h5 file:", h5_folder)
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
        texture_mapping = None
        with open(f"{data_path}/{h5_folder}/texture_mapping.pkl", "rb") as f:
            texture_mapping = pickle.load(f)
        with open(f"{data_path}/{h5_folder}/goal.pkl", "rb") as f:
            goals.append(pickle.load(f))

        mod_pcd_list = []
        sort_name_ids = copy.deepcopy(name_ids)
        sort_name_ids.sort(key=lambda x: x[1])
        for obj_id in goals[0]["obj_ids"]:
            for index, name_id in enumerate(sort_name_ids):
                if name_id[1] == obj_id:
                    mod_pcd_list.append(pcd_list[index])
        
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
        if pattern_name == "line":
            assert len(goals) == 1

        # init region_sampler
        resolution = 0.01
        pix_padding = 1  # padding for clearance
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.0]])  # (height, width, depth)
        region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
        region_sampler.load_from_pcds(pcd_list, name_ids, mask_mode="convex_hull")
        init_pose = region_sampler.get_object_poses()    
        prompt_init = "<round>\nAssume you are a language-based task planner. "
        prompt_init += "Follow the examples we provide. You should strictly adhere to our format.\n"
        prompt_init += "Object_id of the objects in the scene are: ["
        for _, id in name_ids:
            prompt_init += str(id) + ", "
        prompt_init += "]"
        prompt_init += " for ["   
        for name,_ in name_ids:
            prompt_init += '"' + name + '", '
        prompt_init += "].\nAnd correspondingly colors of the objects in the scene are:  ["
        for name,_ in name_ids:
            prompt_init += '"' + texture_mapping[name] + '", '            
        prompt_init += "]. "             
        prompt_init += "\nAnd correspondingly initial poses of the objects in the scene are:  ["
        for entry in init_pose:
            prompt_init += "["
            for pose_entry in init_pose[entry]:
                prompt_init += str(round(pose_entry, 4)) + ", "    
            prompt_init += "], "
        prompt_init += "]. "             
        prompt_init = prompt_init.replace(", ].", "].")
        
        prompt_init += "Poses are in the form [tx, ty, tz, rx, ry, rz]. First three indices correspond to translation and rest three to rotation.\n"
        prompt_init += "Based on the given scene, you should plan the sequence of objects like this to make the rearrangement.\n"
        prompt_init += "The rearrangement when executing will be performed in the order in which you list each obj_id and its corresponding goal pose.\n"
        prompt_init += "The initial pose you are provided with and goal pose you need to predict are both specified in camera coordinates.\n"
        prompt_init += "The region bounds (min-max X,Y,Z) for the rearrangement are [[-0.2, 0.2], [-0.4, 0.4], [0.0, 0.5]], again in camera coordinates only.\n"
        if pattern_name == "dinner":
            prompt_init += "Observe that the user query here is about performing a dinner rearrangement which is a composition of two or three sub rearrangements (uniform, tower, spatial).\n"
            prompt_init += "Hence, your answer needs to have all the object ids and their goal_poses from the given two or three sub rearrangement goals in the user query.\n"
            prompt_init += "For example, if the Query has two sub-goals with pattern:uniform and pattern:spatial, your answer should involve a union of object ids from the given two sub-patterns.\n"

        llm_goal = copy.deepcopy(goals[0])
        if "anchor_id" in llm_goal:
            del llm_goal["anchor_id"] 
        if "anchor_ids" in llm_goal:
            del llm_goal["anchor_ids"]

        prompt_init += "<user>\nQuery: '" + str(llm_goal).replace("'", '"') + "'.\n</user>"
        prompt_init += '\n<assistant>\nAnswer: {"obj_ids": ['

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
        plan_success, action_list = sampling_planner.plan(L, algo="mcts", prior_dict=PATTERN_DICT, debug=True, max_iter=10000, seed=0, is_virtual=False)
        for action in action_list:
                prompt_init += str(int(action["obj_id"])) + ", "
        prompt_init += '], "goal_poses": ['
        for action in action_list:
                prompt_init += "["
                for gpose in action["new_pose"]:
                    prompt_init += str(np.round(gpose, 4)) + ", "    
                prompt_init += "], "
                prompt_init = prompt_init.replace(", ], ", "], ")
        prompt_init += "]}"
        prompt_init = prompt_init.replace(", ]}", "]}")
        prompt_init = prompt_init.replace(", ]]", "]]")
        prompt_init += "\n</assistant>\n</round>\n"
        print("\n" + prompt_init + "========================================================\n") 
        
            
        region_sampler.set_object_poses(init_pose)
        for step in action_list:
            region_sampler.set_object_pose(step["obj_id"], step["new_pose"])

        # Step 4. Calculate Success Rate
        overall_status = True
        for goal in goals:
            obj_poses = {}
            for entry in action_list:
                if entry["obj_id"] in goal["obj_ids"]:
                    obj_poses[entry['obj_id']] = (entry["new_pose"])
            pattern_info = {"threshold": 0.05}
            pattern_info["obj_ids"] = goal["obj_ids"]
            if "spatial_label" in goal:
                pattern_info["spatial_label"] = goal["spatial_label"]
            pattern_status = PATTERN_DICT[goal["type"].split(
                ":")[-1]].check(obj_poses=obj_poses, pattern_info=pattern_info)
            not_collision = not region_sampler.check_collision(goal["obj_ids"])
            if goal["type"] == "pattern:tower":
                not_collision = True
            status = pattern_status and not_collision
            overall_status = overall_status and status
            print(f"Goal type: {goal['type']}, Pattern Status: {pattern_status}; Collision Status: {not not_collision}; Success: {status}")
        
        if overall_status:
            counter += 1
            prompt_full += prompt_init
            if counter == 5:
                break
            mcts_success_result[h5_folder] = {"success_rate": 1, "pattern_status": pattern_status, "not_collision": not_collision}
            mcts_success_rate.append(1)
        else:
            mcts_success_result[h5_folder] = {"success_rate": 0, "pattern_status": pattern_status, "not_collision": not_collision}
            mcts_success_rate.append(0)
            failures.append({"File": h5_folder, "Pattern Status": pattern_status, "Collision Status": not not_collision})
    prompt_full += "\n</root>"
    fp = open(f"lgmcts/prompts/lfsp_sformer_prompt_{pattern_name}.txt", "w")
    fp.write(prompt_full)
    fp.close()
    for fail_case in failures:
        print(fail_case)
    mcts_success_rate = np.array(mcts_success_rate)
    mcts_success_result["success_rate"] = np.mean(mcts_success_rate)*100
    # with open(f"{res_path}/mcts_success_result_{start}_to_{end}.pkl", "wb") as f:
    #     pickle.dump(mcts_success_result, f)
    print("MCTS Success Rate:", np.mean(mcts_success_rate)*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--res_path", type=str, default=None, help="Path to the prompt")
    parser.add_argument("--pattern", type=str, default="line", help="Pattern")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
    args = parser.parse_args()

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    args.data_path = os.path.join(root_path, f"output/lfsp/eval_single_pattern/{args.pattern}-pcd-objs")
    args.res_path = os.path.join(root_path, f"output/lfsp/eval_single_pattern/res-{args.pattern}-pcd-objs")
    eval(args.data_path, args.res_path, args.n_samples, args.start, args.end, pattern_name=args.pattern)