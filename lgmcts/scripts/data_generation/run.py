from __future__ import annotations

import multiprocessing
import os
import pickle
from math import ceil

import hydra
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm

import lgmcts
import lgmcts.utils.file_utils as U

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
    save_path = U.f_join(save_path, task_name)
    os.makedirs(save_path, exist_ok=True)

    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)
    n_generated = 0

    num_tried_this_seed = 0

    env = lgmcts.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed
    )
    task = env.task
    oracle_fn = task.oracle(env)

    metadata = {
        "n_steps_min": 9999999,
        "n_steps_max": 0,
        "n_steps_mean": 0,
        "seed_min": 9999999,
        "seed_max": 0,
    }
    while True:
        try:
            env.seed(seed + n_generated)
            num_tried_this_seed += 1

            obs_cache = []
            action_cache = []

            obs = env.reset()
            obs_cache.append(obs)
            elapsed_steps = 0
            meta, prompt, prompt_assets = env.meta_info, env.prompt, env.prompt_assets

            oracle_failed = False
            for _ in range(task.oracle_max_steps):
                oracle_action = oracle_fn.act(obs)
                if oracle_action is None:
                    print("WARNING: no oracle action, skip!")
                    oracle_failed = True
                    break
                # clip action
                oracle_action = {
                    k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                    for k, v in oracle_action.items()
                }
                obs, _, done, __, info = env.step(action=oracle_action, skip_oracle=False)
                obs_cache.append(obs)
                action_cache.append(oracle_action)
                elapsed_steps += 1
                if done:
                    break
            if oracle_failed:
                seed += 1
                num_tried_this_seed = 0
                continue
        except ValueError as e:
            print(e)
            seed += 1
            num_tried_this_seed = 0
            continue
        assert len(obs_cache) == len(action_cache) + 1 == elapsed_steps + 1
        if success_only and not info["success"]:
            if num_tried_this_seed >= MAX_TRIES_PER_SEED:
                seed += 1
                num_tried_this_seed = 0
            continue

        traj_save_path = U.f_join(save_path, f"{n_generated:0{num_save_digits}d}")
        os.makedirs(traj_save_path, exist_ok=True)
        obs = U.stack_sequence_fields(obs_cache)
        action = U.stack_sequence_fields(action_cache)
        assert U.get_batch_size(obs) == U.get_batch_size(action) + 1, "INTERNAL"
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
        with open(U.f_join(traj_save_path, "action.pkl"), "wb") as f:
            pickle.dump(action, f)

        trajectory = {
            **meta,
            "prompt": prompt,
            "prompt_assets": prompt_assets,
            "steps": elapsed_steps,
            "success": info["success"],
            "failure": info["failure"],
        }
        with open(U.f_join(traj_save_path, "trajectory.pkl"), "wb") as fp:
            pickle.dump(trajectory, fp)

        # update metadata
        metadata["n_steps_min"] = min(metadata["n_steps_min"], elapsed_steps)
        metadata["n_steps_max"] = max(metadata["n_steps_max"], elapsed_steps)
        metadata["n_steps_mean"] += elapsed_steps
        metadata["seed_min"] = min(metadata["seed_min"], env.task.seed)
        metadata["seed_max"] = max(metadata["seed_max"], env.task.seed)

        n_generated += 1
        num_tried_this_seed = 0
        tbar.update(1)

        if n_generated >= num_episodes:
            break
    tbar.close()
    metadata["n_steps_mean"] /= n_generated
    with open(U.f_join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)


def generate_data_for_one_task(kwargs):
    _generate_data_for_one_task(**kwargs)