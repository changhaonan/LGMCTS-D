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

def eval_offline(dataset_path: str, start: int, end: int, mask_mode: str, n_epoches: int = 10, debug: bool = True):
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
    cam2world = env.agent_cams["top"]["transform"]
    bounds = np.array([[-0.2, 0.2], [-0.4, 0.4], [0.0, 0.5]])  # bounds in camera coordinate
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    prompt_generator = PromptGenerator(env.rng)
    sucess_count = 0
    plan_success_count = 0
    exe_success_count = 0
    action_step_count = 0
    failed_count = 0
    # LLM parsing
    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    checkpoint_list = checkpoint_list[start:end]
    n_epoches = 0
    missed = 0
    with open(f"{dataset_path}/llm_res_{start}_to_{end}.pkl","rb") as fp:
        llm_result = pickle.load(fp)
    for i in range(len(llm_result)):
        if llm_result[i] is None:
            missed += 1
            print("========>>> Missed:", missed)
            continue
        # Step 1. init the env from dataset
        n_epoches += 1
        print("Current Epoch:", n_epoches)
        env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # load from dataset
        checkpoint_path = os.path.join(dataset_path, checkpoint_list[i])
        env.load_checkpoint(checkpoint_path)
        prompt_generator.prompt = task.prompt
        region_sampler.load_env(mask_mode=mask_mode, env=env)
        env.prepare()
        action_list = llm_result[i]
        for step in action_list:
            # position
            pose0_position = cam2world[:3, :3] @ step["old_pose"][:3] + cam2world[:3, 3]
            pose0_position[2] = 0.0
            pose1_position = cam2world[:3, :3] @ step["new_pose"][:3] + cam2world[:3, 3]
            pose1_position[2] = 0.05
            # FIXME: rotation currently is not working
            # rotation, only contains z
            rot_z = R.from_euler("z", step["old_pose"][5]).as_matrix()
            pose0_rotation = R.from_matrix(cam2world[:3, :3] @ rot_z).as_quat()
            rot_z = R.from_euler("z", step["new_pose"][5]).as_matrix()
            pose1_rotation = R.from_matrix(cam2world[:3, :3] @ rot_z).as_quat()
            
            # DEBUG
            pose0_rotation = np.array([0, 0, 0, 1])
            pose1_rotation = np.array([0, 0, 0, 1])
            action = {
                "pose0_position": pose0_position,
                "pose0_rotation": pose0_rotation,
                "pose1_position": pose1_position,
                "pose1_rotation": pose1_rotation,
            }
            # execute action
            env.step(action)

        # Step 4. evaluate the result
        action_step_count += len(action_list)
        exe_result = task.check_success(obj_poses=env.get_obj_poses(), flip_xy=True)
        overall_success = exe_result.success
        print(f"Execute sucess: {exe_result.success}; Success: {overall_success}")
        if exe_result.success:
            exe_success_count += 1
        if overall_success:
            sucess_count += 1

        if debug:
            prompt_generator.render(append=" [succes]" if overall_success else " [fail]")

        print("----------- Current Result -----------")
        print(f"Success rate: {float(sucess_count) / float(i + 1)}")
        print(f"Execute success rate: {float(exe_success_count) / float(i + 1)}")
        print(f"Average action steps: {float(action_step_count) / float(i + 1)}")
    # average result
    print("----------- Final Result -----------")
    print(f"Success rate: {float(sucess_count) / float(n_epoches)}")
    print(f"Execute success rate: {float(exe_success_count) / float(n_epoches)}")
    print(f"Average action steps: {float(action_step_count) / float(n_epoches)}")
    # log result
    result_dict = {
        "start": start,
        "end": end,
        "success_rate": float(sucess_count) / float(n_epoches),
        "plan_success_rate": float(plan_success_count) / float(n_epoches),
        "exe_success_rate": float(exe_success_count) / float(n_epoches),
        "average_action_steps": float(action_step_count) / float(n_epoches),
    }
    save_name = f"score_{start}_to_{end}.pkl"
    with open(f"{dataset_path}/{save_name}","wb") as fp:
        pickle.dump(result_dict, fp)
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
    seed = args.seed
    # manually set
    # args.debug = True
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/lfsp/elgr/struct_rearrange_{seed}"
    eval_offline(dataset_path=dataset_path, start=args.start, end=args.end, mask_mode=args.mask_mode, debug=args.debug)
