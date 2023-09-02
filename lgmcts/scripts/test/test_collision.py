"""Test the collision behavior"""
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


def test_collision(dataset_path: str, method: str, n_samples: int = 10, n_epoches: int = 10, debug: bool = True):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    resolution = 0.002
    n_samples = 5
    num_save_digits = 6
    env = lgmcts.make(
        task_name=task_name, 
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name], 
        modalities=["rgb", "segm", "depth"], 
        seed=0, 
        debug=debug, 
        display_debug_window=debug,
        hide_arm_rgb=True)
    task = env.task

    region_sampler = Region2DSamplerLGMCTS(resolution, env)
    prompt_generator = PromptGenerator(env.rng)
    sampling_planner = SamplingPlanner(region_sampler, n_samples=n_samples)  # bind sampler
    sucess_count = 0

    checkpoint_list = list(filter(lambda f: f.endswith(".pkl"), os.listdir(dataset_path)))
    checkpoint_list.sort()
    
    ## Test the collision
    i = 0
    print(f"==== Episode {i} ====")
    ## Step 1. init the env from dataset
    env.reset()
    prompt_generator.reset()
    region_sampler.reset()
    # load from dataset
    checkpoint_path = os.path.join(dataset_path, checkpoint_list[i])
    env.load_checkpoint(checkpoint_path)
    prompt_generator.prompt = task.prompt
    region_sampler.load_objs_from_env(env, mask_mode="raw_mask")
    # DEBUG
    # region_sampler.visualize()
    if debug:
        prompt_generator.render()
        region_sampler.visualize()

    # close
    env.close()
    prompt_generator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--method", type=str, default="mcts", help="Method to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--n_epoches", type=int, default=10, help="Number of epoches")
    # parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        dataset_path = f"{root_path}/output/struct_rearrange"
    test_collision(dataset_path=dataset_path, method=args.method, n_samples=args.n_samples, n_epoches=args.n_epoches, debug=True)