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
from lgmcts.components.prompt import PromptGenerator

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
    # prepare path
    save_path = U.f_join(save_path, task_name)
    save_path = os.path.join(save_path, "circle/result")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "batch300"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "index"), exist_ok=True)

    n_generated = 0
    num_tried_this_seed = 0
    tbar = tqdm(total=num_episodes, desc=task_name, leave=True)

    # init
    env = lgmcts.make(
        task_name=task_name, 
        task_kwargs=task_kwargs, 
        modalities=modalities, 
        seed=seed, 
        debug=False, 
        display_debug_window=False,
        hide_arm_rgb=True,
    )
    task = env.task
    prompt_generator = PromptGenerator(env.rng)
    export_file_list = []

    while True:
        try:
            env.set_seed(seed + n_generated)
            num_tried_this_seed += 1
            obs_cache = []

            step_t = 0
            # reset
            env.reset()
            prompt_generator.reset()

            # generate goal
            prompt_str, obs = task.gen_goal_config(env, prompt_generator)
            obs_cache.append(obs)

            # generate start    
            obs = task.gen_start_config(env)
            goal_spec = task.gen_goal_spec(env)
            obs_cache.append(obs)

            step_t += 1
        except Exception as e:
            print(e)
            seed += 1
            num_tried_this_seed = 0
            continue
        
        ## Process output data
        obs = U.stack_sequence_fields(obs_cache)
        # save data into hdf5, which is required by the data loader
        view = "top"
        export_file_name = f"batch300/data_{n_generated:08}.h5"
        export_file_list.append(export_file_name)
        with h5py.File(U.f_join(save_path, export_file_name), 'w') as f:
            # rgb
            rgb = obs.pop("rgb")
            rgb = rearrange(rgb[view], "t c h w -> t h w c")
            f.create_dataset("rgb", data=rgb)
            # seg
            seg = obs.pop("segm")
            seg = seg[view][:, :, :, None]  # Append a new dimension
            f.create_dataset("seg", data=seg)
            # depth
            depth = obs.pop("depth")
            depth = rearrange(depth[view], "t c h w -> t h w c")
            f.create_dataset("depth", data=depth)
            # depth_min & depth_max
            depth_min = np.min(depth) * np.ones([2, 1], dtype=np.float32)
            depth_max = np.max(depth) * np.ones([2, 1], dtype=np.float32)
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
            cam_pose = cam_pose[None, :, :].repeat(step_t+1, axis=0)
            f.create_dataset("camera_view", data=cam_pose)
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
            f.create_dataset("goal_specification", data=json.dumps(goal_spec))
                
        n_generated += 1
        num_tried_this_seed = 0
        tbar.update(1)

        if n_generated >= num_episodes:
            break
    tbar.close()

    # generate index
    indices_all = []
    for i, file in enumerate(export_file_list):
        indices_all.append((file, i))
    with open(os.path.join(save_path, "index/test_arrangement_indices_file_all.txt"), "w") as f:
        json.dump(indices_all, f)

    # generate vocabulary
    type_vocab = task.gen_type_vocabs()
    with open(os.path.join(save_path, "../type_vocabs_coarse.json"), "w") as f:
        json.dump(type_vocab, f)


if __name__ == '__main__':
    task_name = "struct_rearrange"
    _generate_data_for_one_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        num_episodes=10,
        save_path="/Users/haonanchang/Projects/LGMCTS-D/output/struct_diffusion",
        num_save_digits=8,
        seed=0,
    )