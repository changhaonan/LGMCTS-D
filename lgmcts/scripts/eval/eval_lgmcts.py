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

# Rigid sample Data


# Eval method

def eval_offline(dataset_path: str, method: str, mask_mode: str, n_samples: int = 10, n_epoches: int = 10, debug: bool = True):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    resolution = 0.01
    pix_padding = 1  # padding for clearance
    n_samples = 5
    num_save_digits = 6
    use_gt_pose = False  # if directly use the pose from dataset
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
    action_step_count = 0
    # LLM parsing
    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    n_epoches = min(n_epoches, len(checkpoint_list))
    use_llm = False
    run_llm = False
    encode_ids_to_llm = False
    # Generate goals using llm and object selector
    prompt_goals = gen_prompt_goal_from_llm(dataset_path, n_epoches, checkpoint_list, use_llm=use_llm,
                                            run_llm=run_llm, encode_ids_to_llm=encode_ids_to_llm, num_save_digits=num_save_digits, debug=debug)

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
        region_sampler.load_env(mask_mode=mask_mode, env=env)
        init_pose = region_sampler.get_object_poses()
        # DEBUG
        if debug:
            # region_sampler.visualize()
            prompt_generator.render()
            ##
            print(env.obj_ids)

        # Step 2. build a sampler based on the goal (from goal is cheat, we want to from LLM in the future)
        if prompt_goals is None:
            goals = task.goals
        else:
            goals = prompt_goals[i]
        # pattern remapping
        if use_gt_pose:
            with open(os.path.join(dataset_path, checkpoint_list[i].replace("checkpoint_", "goal_spec_")), "rb") as f:
                goal_spec = pickle.load(f)
            goals = REMAPPING_PATTERN_DICT["rigid"].parse_goal(goals=goals, goal_spec=goal_spec, region_sampler=region_sampler, env=env)
            region_sampler.set_object_poses(init_pose)  # reset region sampler
        L = []
        for goal in goals:
            goal_obj_ids = goal["obj_ids"]
            goal_pattern = goal["type"].split(":")[-1]
            print(f"Goal: {goal_pattern}; {goal_obj_ids}")

            for _i, goal_obj_id in enumerate(goal_obj_ids):
                sample_info = goal.get("sample_info", {})
                if goal_pattern == "spatial":
                    # spatial only sample the second obj
                    if _i == 0:
                        continue
                    else:
                        sample_info["spatial_label"] = goal["spatial_label"]
                        sample_info["ordered"] = True
                sample_data = SampleData(goal_pattern, goal_obj_id, goal["obj_ids"], {}, sample_info)
                L.append(sample_data)

        # Step 3. generate & exectue plan
        action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug, max_iter=20000, seed=1)
        print("Plan finished!")
        env.prepare()
        for step in action_list:
            # assemble action
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
        print(f"Success: {exe_result.success}")
        if exe_result.success:
            sucess_count += 1

        if debug:
            prompt_generator.render(append=" [succes]" if exe_result.success else " [fail]")

        print("----------- Current Result -----------")
        print(f"Success rate: {float(sucess_count) / float(i + 1)}")
        print(f"Average action steps: {float(action_step_count) / float(i + 1)}")

    # average result
    print("----------- Final Result -----------")
    print(f"Success rate: {float(sucess_count) / float(n_epoches)}")
    print(f"Average action steps: {float(action_step_count) / float(n_epoches)}")
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
    args.debug = True
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/struct_rearrange"
    eval_offline(dataset_path=dataset_path, method=args.method, mask_mode=args.mask_mode,
                 n_samples=args.n_samples, n_epoches=args.n_epoches, debug=args.debug)
