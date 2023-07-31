"""Generate data for struct diffusion"""
from __future__ import annotations

import multiprocessing
import os
import json
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
import lgmcts.utils.misc_utils as utils
from lgmcts import PARTITION_TO_SPECS

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

    n_generated = 0
    num_tried_this_seed = 0
    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)

    # Init env & task
    env = lgmcts.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed
    )
    task = env.task

    while True:
        try:
            env.set_seed(seed + n_generated)
            num_tried_this_seed += 1
            obs_cache = []
            action_cache = []
            goal_specs = []

            # Start-config
            obs = env.reset()
            obs_cache.append(obs)
            elapsed_steps = 0
            meta, prompt, prompt_assets, goal_spec = env.meta_info, env.prompt, env.prompt_assets, env.goal_specification

            # Set to start state
            obs = task.start(env)
            obs_cache.append(obs)

            # Run
        except Exception as e:
            print(e)
            seed += 1
            num_tried_this_seed = 0
            continue

        # traj_save_path = U.f_join(save_path, f"{n_generated:0{num_save_digits}d}")
        # os.makedirs(traj_save_path, exist_ok=True)
        obs = U.stack_sequence_fields(obs_cache)
        # rgb = obs.pop("rgb")
        # views = sorted(rgb.keys())
        # for view in views:
        #     frames = rgb[view]
        #     frames = rearrange(frames, "t c h w -> t h w c")
        #     rgb_per_view_save_path = U.f_join(traj_save_path, f"rgb_{view}")
        #     os.makedirs(rgb_per_view_save_path, exist_ok=True)
        #     # loop over time dimension to save as jpg using PIL.Image
        #     for i, frame in enumerate(frames):
        #         img = Image.fromarray(frame, mode="RGB")
        #         img.save(U.f_join(rgb_per_view_save_path, f"{i}.jpg"))
        # with open(U.f_join(traj_save_path, "obs.pkl"), "wb") as f:
        #     pickle.dump(obs, f)
        
        # save trajectory
        # trajectory = {
        #     **meta,
        #     "prompt": prompt,
        #     "prompt_assets": prompt_assets,
        #     "steps": elapsed_steps,
        #     # "success": info["success"],
        #     # "failure": info["failure"],
        # }
        # with open(U.f_join(traj_save_path, "trajectory.pkl"), "wb") as fp:
        #     pickle.dump(trajectory, fp)

        # save data into hdf5, which is required by the data loader
        view = "top"
        with h5py.File(U.f_join(save_path, f"data_{n_generated:08}.h5"), 'w') as f:
            # rgb
            rgb = obs.pop("rgb")
            rgb = rearrange(rgb[view], "t c h w -> t h w c")
            f.create_dataset("rgb", data=rgb)
            # seg
            seg = obs.pop("segm")
            seg = seg[view]
            f.create_dataset("seg", data=seg)
            # depth
            depth = obs.pop("depth")
            depth = rearrange(depth[view], "t c h w -> t h w c")
            f.create_dataset("depth", data=depth)
            # depth_min & depth_max
            depth_min = np.min(depth)
            depth_max = np.max(depth)
            f.create_dataset("depth_min", data=depth_min)
            f.create_dataset("depth_max", data=depth_max)
            # camera related
            intrinsic = np.array(env.agent_cams[view]["intrinsics"]).reshape(3, 3)
            f.create_dataset("cam_intrinsics", data=intrinsic)
            image_size = env.agent_cams[view]["image_size"]
            f.create_dataset("cam_width", data=image_size[0])
            f.create_dataset("cam_height", data=image_size[1])
            cam_position = env.agent_cams[view]["position"]
            cam_rotation = env.agent_cams[view]["rotation"]
            cam_pose = utils.get_transfroms(cam_position, cam_rotation)
            f.create_dataset("cam_view", data=cam_pose)
            # objs related
            poses = obs.pop("poses")
            poses = poses[view].transpose(1, 0, 2)
            for i, obj_id in enumerate(env.obj_ids["rigid"]):
                obj_code = f"object_{i:02}"
                f.create_dataset(f"id_{obj_code}", data=obj_id)
                obj_pose = utils.get_transforms_batch(
                    poses[i, :, :3], poses[i, :, 3:]
                )
                f.create_dataset(obj_code, data=obj_pose)
            # goal spec
            f.create_dataset("goal_spec", data=json.dumps(goal_spec))
                
        n_generated += 1
        num_tried_this_seed = 0
        tbar.update(1)

        if n_generated >= num_episodes:
            break
    tbar.close()


if __name__ == '__main__':
    task_name = "structure_rearrange"
    _generate_data_for_one_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        num_episodes=10,
        success_only=True,
        save_path="/Users/haonanchang/Projects/LGMCTS-D/output/struct_diffusion",
        num_save_digits=8,
        seed=0,
    )