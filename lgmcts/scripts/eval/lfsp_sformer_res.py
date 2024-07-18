"""Generate results by LFSP on the Structformer dataset"""
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
    prompts = []
    init_pose_list = []
    for iter in tqdm.tqdm(range(start, end)):
        h5_folder = h5_folders[iter]
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
        init_pose_list.append(init_pose)
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
        if pattern == "dinner":
            prompt_init += "Observe that the user query here is about performing a dinner rearrangement which is a composition of two or three sub rearrangements (uniform, tower, spatial).\n"
            prompt_init += "Hence, your answer needs to have all the object ids and their goal_poses from the given two or three sub rearrangement goals in the user query.\n"
        # prompt_init += "Make sure that obj_ids in the user query and your answer are same. Your job is find the order for the action sequence as I mentioned before.\n"
        # prompt_init += f"You are making a {pattern_name} pattern rearrangement here.\n"
        llm_goal = copy.deepcopy(goals)
        if "anchor_id" in llm_goal:
            del llm_goal["anchor_id"] 
        if "anchor_ids" in llm_goal:
            del llm_goal["anchor_ids"]
        prompt_init += "<user>\nQuery: '" + str(llm_goal).replace("'", '"') + "'.\n</user>"
        prompts.append(prompt_init)
    

    prompt_db = open(f"lgmcts/prompts/lfsp_sformer_prompt_{pattern_name}.txt", "r").read()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_path)))), "lgmcts", "conf", "api_key.pkl"), "rb") as fp:
        api_keys = pickle.load(fp)
    api_key = random.choice(api_keys)  
    chatgpt = ChatGPTAPI(model="gpt-4", api_key=api_key, db=prompt_db)
    ret = chatgpt.chat(str_msg=prompts)
    llm_result = ret[0]
    action_lists = []
    missed = 0
    with open(f"{data_path}/raw_llm_{pattern_name}_res_{start}_to_{end}.pkl","wb") as fp:
        pickle.dump(llm_result, fp)
    for scene_id, entry in enumerate(llm_result):
        try:
            entry = entry.replace("Answer: ", "").replace("\n","").replace("<","").replace("/","").replace(">", "").replace("assistant", "")
            import json 
            entry = json.loads(entry)
            action_list = []
            for obj_id, obj_pose in zip(entry["obj_ids"], entry["goal_poses"]):
                action_list.append({"scene_id" : h5_folders[scene_id + start], "obj_id": obj_id, "new_pose" : np.array(obj_pose, dtype=np.float64), "old_pose": init_pose_list[scene_id][obj_id]})
            action_lists.append(action_list)
        except:
            missed += 1
            print(">>>>>Missed:", missed, "scene index:", scene_id)
    if end == -1:
        save_name = "llm_{pattern_name}_res_full.pkl"
    else:
        save_name = f"llm_{pattern_name}_res_{start}_to_{end}.pkl"
    with open(f"{data_path}/{save_name}","wb") as fp:
        pickle.dump(action_lists, fp)
    print("LLM_RESULT LEN:", len(action_lists))

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
