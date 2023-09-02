"""Evaluate the performace of lgmcts system"""
from __future__ import annotations
import os
import time
import pickle
import lgmcts
import argparse
import numpy as np
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as misc_utils
import lgmcts.utils.pybullet_utils as pb_utils
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.components.obj_selector import ObjectSelector
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData


## Eval method

def eval_offline(dataset_path: str, method: str, mask_mode: str, n_samples: int = 10, n_epoches: int = 10, debug: bool = True):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    resolution = 0.01
    pix_padding = 2  # padding for clearance
    n_samples = 5
    num_save_digits = 6
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

    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, env)
    prompt_generator = PromptGenerator(env.rng)
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)  # bind sampler
    sucess_count = 0

    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    for i in range(min(n_epoches, len(checkpoint_list))):
        print(f"==== Episode {i} ====")
        ## Step 1. init the env from dataset
        env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # load from dataset
        checkpoint_path = os.path.join(dataset_path, checkpoint_list[i])
        env.load_checkpoint(checkpoint_path)
        prompt_generator.prompt = task.prompt
        region_sampler.load_objs_from_env(env, mask_mode=mask_mode)
        # DEBUG
        if debug:
            region_sampler.visualize()
            prompt_generator.render()

        ## Step 2. build a sampler based on the goal (from goal is cheat, we want to from LLM in the future)
        goals = task.goals
        L = []
        for goal in goals:
            goal_obj_ids = goal["obj_ids"]
            for goal_obj_id in goal_obj_ids:
                sample_data = SampleData(goal["type"].split(":")[-1], goal_obj_id, goal["obj_ids"], {})
                L.append(sample_data)
        
        ## Step 3. generate & exectue plan
        action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug)

        for step in action_list:
            # assemble action
            action = {
                "pose0_position": step["old_pose"][:3],
                "pose0_rotation": step["old_pose"][3:],
                "pose1_position": step["new_pose"][:3],
                "pose1_rotation": step["new_pose"][3:],
            }
            # execute action
            env.step(action)

        ## Step 4. evaluate the result
        exe_result = task.check_success(obj_poses=env.get_obj_poses())
        print(f"Success: {exe_result.success}")
        if exe_result.success:
            sucess_count += 1
        
        if debug:
            prompt_generator.render(append=" [succes]" if exe_result.success else " [fail]")
            time.sleep(3.0)  # stop a while for eyeballing

    # average result
    print(f"Success rate: {float(sucess_count) / float(n_epoches)}")
    # close
    env.close()
    prompt_generator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    parser.add_argument("--mask_mode", type=str, default="convex_hull", help="Mask mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/struct_rearrange"
    eval_offline(dataset_path=dataset_path, method=args.method, mask_mode=args.mask_mode, n_samples=args.n_samples, n_epoches=args.n_epoches, debug=args.debug)