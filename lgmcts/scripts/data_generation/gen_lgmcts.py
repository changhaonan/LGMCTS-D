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

MAX_TRIES_PER_SEED = 999

def _generate_data_for_one_task(
    task_name: str,
    task_kwargs: dict | None,
    modalities,
    num_episodes: int,
    success_only: bool,
    save_path: str,
    num_save_digits: int,
    seed: int | None = None,
):
    ## Init
    save_path = U.f_join(save_path, task_name)
    os.makedirs(save_path, exist_ok=True)
    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)
    n_generated = 0
    num_tried_this_seed = 0
    # Init task, env, prompt_generator
    env = lgmcts.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed
    )
    task = env.task
    prompt_generator = PromptGenerator(env.rng)

    ## Run generation
    while True:
        try:
            env.set_seed(seed + n_generated)
            num_tried_this_seed += 1
            obs_cache = []
            goal_specs = []

            ## Epoch process
            env.reset()
            prompt_generator.reset()

            # Generate goal first, then generate start based on goal setup
            prompt_str, goal_obs = task.gen_goal_config(env, prompt_generator)
            start_obs = task.gen_start_config(env)
            obs_cache = [start_obs, goal_obs]
            
            # Run
        except Exception as e:
            print(e)
            seed += 1
            num_tried_this_seed = 0
            continue
        
        ## Save data for this epoch
        traj_save_path = U.f_join(save_path, f"{n_generated:0{num_save_digits}d}")
        os.makedirs(traj_save_path, exist_ok=True)
        obs = U.stack_sequence_fields(obs_cache)
        rgb = obs.pop("rgb")
        views = sorted(rgb.keys())
        for view in views:
            frames = rgb[view]
            frames = rearrange(frames, "t c h w -> t h w c")
            rgb_per_view_save_path = U.f_join(traj_save_path, f"rgb_{view}")
            os.makedirs(rgb_per_view_save_path, exist_ok=True)
            # loop over time dimension to save as jpg using PIL.Image
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame, mode="RGB")
                img.save(U.f_join(rgb_per_view_save_path, f"{i}.jpg"))
        with open(U.f_join(traj_save_path, "obs.pkl"), "wb") as f:
            pickle.dump(obs, f)
        with open(U.f_join(traj_save_path, "prompt.txt"), "wb") as f:
            f.write(prompt_str.encode("utf-8"))

        # Update counter
        n_generated += 1
        num_tried_this_seed = 0
        tbar.update(1)
        if n_generated >= num_episodes:
            break
    
    tbar.close()


if __name__ == '__main__':
    task_name = "struct_rearrange"
    _generate_data_for_one_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm"],
        num_episodes=10,
        success_only=True,
        save_path="/Users/haonanchang/Projects/LGMCTS-D/output",
        num_save_digits=6,
        seed=0,
    )