"""Evaluate the performace of lgmcts system"""
from __future__ import annotations
import os
import copy
import time
import pickle
import lgmcts
import argparse
import numpy as np
import json
import ast
from scipy.spatial.transform import Rotation as R
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as misc_utils
import lgmcts.utils.pybullet_utils as pb_utils
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.components.obj_selector import ObjectSelector
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.components.semantic_patterns import REMAPPING_PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
from lgmcts.scripts.data_generation.llm_parse import gen_prompt_goal_from_llm
from lgmcts.env import seed
from lgmcts.components.llm_chatgpt import ChatGPTAPI
import random
# Rigid sample Data


# Eval method

def eval_offline(dataset_path: str, start: int, end: int, mask_mode: str, debug: bool = True, seed: int = 0):
    """Eval from newly generated scene"""
    task_name = f"struct_rearrange_{seed}"
    resolution = 0.01
    pix_padding = 1  # padding for clearance
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=(not debug),
    )
    task = env.task
    bounds = np.array([[-0.2, 0.2], [-0.4, 0.4], [0.0, 0.5]])  # bounds in camera coordinate
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    prompt_generator = PromptGenerator(env.rng)

    # LLM parsing    
    checkpoint_list = list(filter(lambda f: f.endswith(".pkl") and f.startswith("checkpoint_"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    if end != -1:
        checkpoint_list = checkpoint_list[start:end]
    prompts = []
    init_pose_list = []
    for i in range(len(checkpoint_list)):
        # Step 1. init the env from dataset
        env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # load from dataset
        checkpoint_path = os.path.join(dataset_path, checkpoint_list[i])
        env.load_checkpoint(checkpoint_path)
        prompt_generator.prompt = task.prompt
        region_sampler.load_env(mask_mode=mask_mode, env=env)
        init_pose = region_sampler.get_object_poses()
        init_pose_list.append(init_pose)
        prompt_init = "Assume you are a language-based task planner. "
        prompt_init += "Follow the examples we provide. You should strictly adhere to our format.\n"
        prompt_init += "Object_id of the objects in the scene are: "
        prompt_init += str(env.obj_ids["rigid"])
        prompt_init += " for ["   
        for entry in env.obj_ids["rigid"]:
            prompt_init += "'" + env.obj_id_reverse_mapping[entry]["obj_name"] + "', "
        prompt_init += "].\nAnd correspondingly colors of the objects in the scene are:  ["
        for entry in env.obj_ids["rigid"]:
            prompt_init += "'" + env.obj_id_reverse_mapping[entry]["texture_name"] + "', "            
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
        
        prompt_init += "<user>\nQuery1: '" + task.prompt + "'."
        prompts.append(prompt_init)
        # print(prompt_init)       


    prompt_db = open(f"{dataset_path.replace('/lfsp', '').replace(f'_{seed}', '_5')}/prompt_example_direct.txt", "r").read()
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(dataset_path))), "lgmcts", "conf", "api_key.pkl"), "rb") as fp:
        api_keys = pickle.load(fp)
    api_key = random.choice(api_keys)  
    # gpt-3.5-turbo-16k-0613 
    chatgpt = ChatGPTAPI(model="gpt-4", api_key=api_key, db=prompt_db)
    ret = chatgpt.chat(str_msg=prompts)
    llm_result = ret[0]
    # print(ret)
    if debug:
        # region_sampler.visualize()
        prompt_generator.render()
        ##
        print(env.obj_ids)
    action_lists = []
    missed = 0
    for scene_id, entry in enumerate(llm_result):
        try:
            entry = entry.replace("Answer1: ", "")
            import json 
            entry = json.loads(entry)
            action_list = []
            for obj_id, obj_pose in zip(entry["obj_ids"], entry["goal_poses"]):
                action_list.append({"obj_id": obj_id, "new_pose" : np.array(obj_pose, dtype=np.float64), "old_pose": init_pose_list[scene_id][obj_id]})
            action_lists.append(action_list)
        except:
            missed += 1
            print(">>>>>Missed:", missed, "scene index:", scene_id)
    if end == -1:
        save_name = "llm_res_full.pkl"
    else:
        save_name = f"llm_res_{start}_to_{end}.pkl"
    with open(f"{dataset_path}/{save_name}","wb") as fp:
        pickle.dump(action_lists, fp)
    print("LLM_RESULT LEN:", len(action_lists))

    # close
    env.close()
    prompt_generator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--start", type=int, default=75, help="Start index")
    parser.add_argument("--end", type=int, default=100, help="End index")
    parser.add_argument("--seed", type=int, default=1, help="scene seed")
    args = parser.parse_args()

    # manually set
    # args.debug = True
    seed = args.seed 
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/lfsp/struct_rearrange_{seed}"
        # root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        # dataset_path = f"{root_path}/output/struct_rearrange_{seed}"
    eval_offline(dataset_path=dataset_path, start=args.start, end=args.end, mask_mode=args.mask_mode,
                 debug=args.debug, seed=seed)
