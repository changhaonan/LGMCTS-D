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
    seed: int | None = None,
):
    # init
    env = lgmcts.make(
        task_name=task_name, 
        task_kwargs=task_kwargs, 
        modalities=modalities, 
        seed=seed, 
        debug=True, 
        display_debug_window=True,
        hide_arm_rgb=True,
    )
    task = env.task
    prompt_generator = PromptGenerator(env.rng)
    obj_selector = ObjectSelector(env.rng)
    prompt_str_list = []
    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)

    print("Generate dataset...")
    for i in range(num_episodes):
        # reset
        env.reset()
        prompt_generator.reset()
        obj_selector.reset()

        # generate goal
        task.gen_goal_config(env, prompt_generator, obj_selector)
        task.gen_start_config(env)
        
        # save
        prompt_str_list.append(task.prompt)
        env.save_checkpoint(os.path.join(save_path, task_name, f"checkpoint_{i:0{num_save_digits}d}.pkl"))
        tbar.update(1)

    tbar.close()
    # save the prompt string list
    with open(os.path.join(save_path, task_name, "prompt_str_list.txt"), "w") as f:
        f.write("\n".join(prompt_str_list))


if __name__ == '__main__':
    task_name = "struct_rearrange"
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    _generate_data_for_one_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm"],
        num_episodes=10,
        save_path=f"{root_path}/output",
        num_save_digits=6,
        seed=0,
    )