"""Evaluate the performace of lgmcts system"""
from __future__ import annotations
import os
import time
import pickle
import lgmcts
import numpy as np
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as misc_utils
import lgmcts.utils.pybullet_utils as pb_utils
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.algorithm import SamplingPlanner, Region2DSamplerLGMCTS, SampleData
from lgmcts.components.patterns import PATTERN_DICT


## Eval method

def eval_offline(dataset_path: str, n_samples: int = 10):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    resolution = 0.01
    n_samples = 5
    num_save_digits = 6

    env = lgmcts.make(
        task_name=task_name, 
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name], 
        modalities=["rgb", "segm", "depth"], 
        seed=0, 
        debug=True, 
        display_debug_window=True,
        hide_arm_rgb=False,
    )
    task = env.task

    region_sampler = Region2DSamplerLGMCTS(resolution, env)
    prompt_generator = PromptGenerator(env.rng)
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)  # bind sampler

    for i in range(10):
        ## Step 1. init the env from dataset
        env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # load from dataset
        checkpoint_path = os.path.join(dataset_path, f"checkpoint_{i:0{num_save_digits}d}.pkl")
        env.load_checkpoint(checkpoint_path)
        # update region sampler
        region_sampler.load_objs_from_env(env)
        region_sampler.visualize()

        ## Step 2. build a sampler based on the goal (from goal is cheat, we want to from LLM in the future)
        goals = task.goals
        L = []
        for goal in goals:
            goal_obj_ids = goal["obj_ids"]
            for goal_obj_id in goal_obj_ids:
                sample_data = SampleData(goal["type"].split(":")[-1], goal_obj_id, goal["obj_ids"], {})
                L.append(sample_data)

        ## Step 3. generate & exectue plan
        action_list = sampling_planner.plan(L, algo="seq", prior_dict=PATTERN_DICT, debug=True)
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
            time.sleep(3.0)
        # test the env & sampler alignment

        ## Step 4. evaluate the result
        exe_result = task.check_success(obj_poses=env.get_obj_poses())
        print(f"==== Episode {i} ====")
        print(f"Success: {exe_result.success}")


if __name__ == "__main__":
    dataset_path = "/Users/haonanchang/Projects/LGMCTS-D/output/struct_rearrange"
    eval_offline(dataset_path)