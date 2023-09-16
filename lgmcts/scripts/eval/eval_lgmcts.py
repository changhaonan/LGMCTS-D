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
# Rigid sample Data


# Eval method

def eval_offline(dataset_path: str, start: int, end: int, method: str, mask_mode: str, n_samples: int = 10, n_epoches: int = 10, debug: bool = True, use_gt_pose: bool = False, use_llm: bool = False, run_llm: bool = False):
    """Eval from newly generated scene"""
    task_name = f"struct_rearrange_{seed}"
    resolution = 0.01
    pix_padding = 1  # padding for clearance
    n_samples = 5
    num_save_digits = 6
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=4,
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
    plan_success_count = 0
    exe_success_count = 0
    action_step_count = 0
    failed_count = 0
    # LLM parsing
    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    checkpoint_list = checkpoint_list[start:end]
    n_epoches = min(n_epoches, len(checkpoint_list))
    use_llm = False
    run_llm = False
    encode_ids_to_llm = False
    # Generate goals using llm and object selector
    prompt_goals = gen_prompt_goal_from_llm(dataset_path, n_epoches, checkpoint_list, use_llm=use_llm,
                                            run_llm=run_llm, encode_ids_to_llm=encode_ids_to_llm, num_save_digits=num_save_digits, debug=debug)
    task_failed = 0
    for i in range(n_epoches):
        try:
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
                goals = copy.deepcopy(task.goals)
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
            plan_success, action_list = sampling_planner.plan(L, algo=method, prior_dict=PATTERN_DICT, debug=debug, max_iter=10000, seed=1)
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
            overall_success = exe_result.success and plan_success
            print(f"Plan sucess: {plan_success}; Execute sucess: {exe_result.success}; Success: {overall_success}")
            if plan_success:
                plan_success_count += 1
            if exe_result.success:
                exe_success_count += 1
            if overall_success:
                sucess_count += 1

            if debug:
                prompt_generator.render(append=" [succes]" if overall_success else " [fail]")

            print("----------- Current Result -----------")

            print(f"Success rate: {float(sucess_count) / float(i + 1)}")
            print(f"Plan success rate: {float(plan_success_count) / float(i + 1)}")
            print(f"Execute success rate: {float(exe_success_count) / float(i + 1)}")
            print(f"Average action steps: {float(action_step_count) / float(i + 1)}")
        except:
            task_failed += 1    
            print("Cannot solve this task!")
    # average result
    print("----------- Final Result -----------")
    print(f"Success rate: {float(sucess_count) / float(n_epoches)}")
    print(f"Plan success rate: {float(plan_success_count) / float(n_epoches)}")
    print(f"Execute success rate: {float(exe_success_count) / float(n_epoches)}")
    print(f"Average action steps: {float(action_step_count) / float(n_epoches)}")
    print(f"Task failed: {task_failed}")
    # log result
    result_dict = {
        "start": start,
        "end": end,
        "success_rate": float(sucess_count) / float(n_epoches),
        "plan_success_rate": float(plan_success_count) / float(n_epoches),
        "exe_success_rate": float(exe_success_count) / float(n_epoches),
        "average_action_steps": float(action_step_count) / float(n_epoches),
        "task_failed": task_failed,
    }
    with open(os.path.join(dataset_path, f"{method}_{mask_mode}_{start}_{end}_{str(use_gt_pose)}_result.json"), "w") as f:
        json.dump(result_dict, f)
    with open(os.path.join(dataset_path, f"{method}_{mask_mode}_{start}_{end}_{str(use_gt_pose)}_goal_check.pkl"), "wb") as f:
        pickle.dump({"llm" : prompt_goals, "sys" : task_goals}, f)
    
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
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=50, help="End index")
    parser.add_argument("--use_gt_pose", action="store_true", help="Use gt pose")
    parser.add_argument("--use_llm", action="store_true", help="Use llm")
    parser.add_argument("--run_llm", action="store_true", help="Run llm")
    args = parser.parse_args()

    # manually set
    # args.debug = True
    args.use_gt_pose = False
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/struct_rearrange_{seed}"
    eval_offline(dataset_path=dataset_path, start=args.start, end=args.end, method=args.method, mask_mode=args.mask_mode,
                 n_samples=args.n_samples, n_epoches=args.n_epoches, debug=args.debug, use_llm=args.use_llm, run_llm=args.run_llm, use_gt_pose=args.use_gt_pose)
