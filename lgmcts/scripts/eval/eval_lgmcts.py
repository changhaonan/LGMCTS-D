"""Evaluate the performace of lgmcts system"""
from __future__ import annotations
import os
import time
import pickle
import lgmcts
import argparse
import numpy as np
import json
import ast
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
from lgmcts.scripts.data_generation.llm_parse import perform_llm_parsing


# Eval method

def eval_offline(dataset_path: str, method: str, mask_mode: str, n_samples: int = 10, n_epoches: int = 10, debug: bool = True):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    resolution = 0.01
    pix_padding = 1  # padding for clearance
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
    cam2world = env.agent_cams["top"]["transform"]
    bounds = np.array([[-0.2, 0.2], [-0.4, 0.4], [0.0, 0.5]])  # bounds in camera coordinate
    region_sampler = Region2DSamplerLGMCTS(resolution, pix_padding, bounds)
    prompt_generator = PromptGenerator(env.rng)
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)  # bind sampler
    sucess_count = 0

    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    n_epoches = min(n_epoches, len(checkpoint_list))
    use_llm = False
    run_llm = False
    prompt_goals = None
    # Generate goals using llm and object selector
    if use_llm:
        if run_llm:
            result = perform_llm_parsing(prompt_bg_file=f"{dataset_path}/prompt_bg.txt",
                                         prompt_str_file=f"{dataset_path}/prompt_str_list.txt", debug=debug)
            res = [ast.literal_eval(r) for r in result]
            with open(os.path.join(os.path.dirname(dataset_path), "prompt", "llm_result.pkl"), "wb") as fp:
                pickle.dump(res, fp)
            obj_selector = ObjectSelector(env.rng)
            obj_selector.parse_llm_result(dataset_path, res, checkpoint_list[:20], len(task.obj_list))
        else:
            with open(os.path.join(dataset_path, "goal.pkl"), "rb") as fp:
                prompt_goals = pickle.load(fp)

    for i in range(n_epoches):
        print(f"==== Episode {i} ====")
        # Step 1. init the env from dataset
        env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # load from dataset
        checkpoint_path = os.path.join(dataset_path, checkpoint_list[i])
        env.load_checkpoint(checkpoint_path)
        prompt_generator.prompt = task.prompt
        region_sampler.load_env(env, mask_mode=mask_mode)
        # DEBUG
        if debug:
            # region_sampler.visualize()
            prompt_generator.render()
            ##
            print(env.obj_ids)

        # Step 2. build a sampler based on the goal (from goal is cheat, we want to from LLM in the future)
        if use_llm:
            goals = prompt_goals[i]
        else:
            goals = task.goals
        L = []
        for goal in goals:
            goal_obj_ids = goal["obj_ids"]
            goal_pattern = goal["type"].split(":")[-1]
            print(f"Goal: {goal_pattern}; {goal_obj_ids}")

            for _i, goal_obj_id in enumerate(goal_obj_ids):
                sample_info = {}
                if goal_pattern == "spatial":
                    # spatial only sample the second obj
                    if _i == 0:
                        continue
                    else:
                        sample_info = {"spatial_label": goal["spatial_label"], "ordered": True}
                sample_data = SampleData(goal_pattern, goal_obj_id, goal["obj_ids"], {}, sample_info)
                L.append(sample_data)

        # Step 3. generate & exectue plan
        action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug)

        env.prepare()
        for step in action_list:
            # assemble action
            pose0_position = cam2world[:3, :3] @ step["old_pose"][:3] + cam2world[:3, 3]
            pose0_position[2] = 0.0
            pose1_position = cam2world[:3, :3] @ step["new_pose"][:3] + cam2world[:3, 3]
            pose1_position[2] = 0.05
            action = {
                "pose0_position": pose0_position,
                "pose0_rotation": step["old_pose"][3:],
                "pose1_position": pose1_position,
                "pose1_rotation": step["new_pose"][3:],
            }
            # execute action
            env.step(action)

        # Step 4. evaluate the result
        exe_result = task.check_success(obj_poses=env.get_obj_poses())
        print(f"Success: {exe_result.success}")
        if exe_result.success:
            sucess_count += 1

        if debug:
            prompt_generator.render(append=" [succes]" if exe_result.success else " [fail]")

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
    eval_offline(dataset_path=dataset_path, method=args.method, mask_mode=args.mask_mode,
                 n_samples=args.n_samples, n_epoches=args.n_epoches, debug=args.debug)
