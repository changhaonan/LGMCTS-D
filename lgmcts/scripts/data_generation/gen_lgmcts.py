"""Generate data for lgmcts tasks"""
from __future__ import annotations

import multiprocessing
import os
import pickle
from math import ceil
import h5py
import hydra
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import argparse

import lgmcts
import lgmcts.utils.file_utils as U
from lgmcts import PARTITION_TO_SPECS
from lgmcts.components.prompt import PromptGenerator
from lgmcts.components.obj_selector import ObjectSelector

MAX_TRIES_PER_SEED = 999


def _generate_data_for_one_task(
    task_name: str,
    task_kwargs: dict | None,
    modalities,
    num_episodes: int,
    save_path: str,
    num_save_digits: int,
    debug: bool,
    seed: int | None = None,
):
    # init
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=task_kwargs,
        modalities=modalities,
        seed=seed,
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=not debug,
    )
    task = env.task
    prompt_generator = PromptGenerator(env.rng)
    obj_selector = ObjectSelector(env.rng)
    prompt_str_list = []
    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)

    # Prepare prompt background
    prompt_bg = "Assume you are a language-based motion planner. You will parse user's requirement into goal configuration and constraints. Follow the examples we provide. You should strictly adhere to our format. \n"
    # obj_id_list = list(range(len(task.obj_list)))
    # obj_name_list = []
    # obj_color_list = []
    # for obj_id in obj_id_list:
    #     obj_name_list.append(task.obj_list[obj_id].name.lower().replace("shapenet_", ""))
    #     obj_color_list.append(task.color_list[obj_id].name.lower().replace("_", " "))
    # prompt_bg += f"Object_id of the objects in the scene are: {obj_id_list} for {obj_name_list}\n"
    # prompt_bg += f"And correspondingly colors of the objects in the scene are:  {obj_color_list}\n"

    with open(os.path.join(save_path, task_name, "prompt_bg.txt"), "w") as f:
        f.write(prompt_bg)

    print("Generate dataset...")
    for i in range(num_episodes):
        print(f"==== Episode {i} ====")
        # reset
        seed = env.rng.integers(0, 100)
        env.set_seed(seed)
        env.reset()
        prompt_generator.reset()
        obj_selector.reset()

        # generate goal
        task.gen_goal_config(env, prompt_generator, obj_selector)
        goal_spec = task.gen_goal_spec(env)
        task.gen_start_config(env)

        # save
        prompt_str_list.append(task.prompt)
        env.save_checkpoint(os.path.join(save_path, task_name, f"checkpoint_{i:0{num_save_digits}d}.pkl"))
        with open(os.path.join(save_path, task_name, f"goal_spec_{i:0{num_save_digits}d}.pkl"), "wb") as f:
            pickle.dump(goal_spec, f)
        tbar.update(1)

    tbar.close()
    # save the prompt string list
    with open(os.path.join(save_path, task_name, "prompt_str_list.txt"), "w") as f:
        f.write("\n".join(prompt_str_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="struct_rearrange")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    _generate_data_for_one_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm"],
        num_episodes=args.num_episodes,
        save_path=f"{root_path}/output",
        num_save_digits=6,
        debug=args.debug,
        seed=0,
    )
