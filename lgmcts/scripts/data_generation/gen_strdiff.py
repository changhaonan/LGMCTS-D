"""Generate data for struct diffusion"""
from __future__ import annotations

import cv2
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
import argparse

import lgmcts
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as utils
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
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=not debug,
    )
    task = env.task
    prompt_generator = PromptGenerator(env.rng)
    obj_selector = ObjectSelector(env.rng)
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
            obj_selector.reset()

            # generate goal
            prompt_str, obs = task.gen_goal_config(env, prompt_generator, obj_selector)
            obs_cache.append(obs)

            # generate start
            obs = task.gen_start_config(env)
            goal_spec = task.gen_goal_spec(env)
            obs_cache.append(obs)

            step_t += 1
        except Exception as e:
            print('strdiff exception:', e)
            seed += 1
            num_tried_this_seed = 0
            continue

        # Process output data
        obs = U.stack_sequence_fields(obs_cache)
        # save data into hdf5, which is required by the data loader
        view = "top"
        export_file_name = f"batch300/data_{n_generated:08}.h5"
        export_file_list.append(export_file_name)
        with h5py.File(U.f_join(save_path, export_file_name), 'w') as f:
            # rgb
            rgb = obs.pop("rgb")
            rgb_tensor = rearrange(rgb[view], "t c h w -> t h w c")
            f.create_dataset("rgb", data=rgb_tensor)
            # seg
            seg = obs.pop("segm")
            seg_tensor = seg[view][:, :, :, None]  # Append a new dimension
            f.create_dataset("seg", data=seg_tensor)
            # depth
            depth = obs.pop("depth")
            depth_tensor = rearrange(depth[view], "t c h w -> t h w c")
            # normalize depth to fit in structFormer
            depth_tensor = depth_tensor * 20.0
            # depth_min & depth_max
            depth_min = np.min(depth_tensor) * np.ones([1,], dtype=np.float32)
            depth_max = np.max(depth_tensor) * np.ones([1,], dtype=np.float32)
            f.create_dataset("depth_min", data=depth_min)
            f.create_dataset("depth_max", data=depth_max)
            # normalize depth
            depth_tensor = (depth_tensor - depth_min) / (depth_max - depth_min) * 20000.0
            f.create_dataset("depth", data=depth_tensor)
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
            # strdiff-sytle camera information
            f_x = intrinsic[0, 0]
            proj_fov = np.degrees(2 * np.arctan(image_size[1] / (2 * f_x)))
            f.create_dataset("proj_fov", data=proj_fov)
            f.create_dataset("proj_near", data=0.5)
            f.create_dataset("proj_far", data=5.0)
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

            # DEBUG
            # print(rgb.shape)
            # check sem is not empty
            for i, obj_id in enumerate(env.obj_ids["rigid"]):
                obj_mask = seg_tensor[0, :, :, 0] == obj_id
                if np.sum(obj_mask) == 0:
                    cv2.imshow("rgb", rgb_tensor[0, :, :, :])
                    cv2.imshow("seg", seg_tensor[0, :, :, 0] / (np.max(seg_tensor[0, :, :, 0]) + 1e-6) * 255)
                    cv2.waitKey(0)
                    assert False, f"Object {obj_id} is not in the image"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="struct_rearrange")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    _generate_data_for_one_task(
        args.task_name,
        PARTITION_TO_SPECS["train"][args.task_name],
        modalities=["rgb", "segm", "depth"],
        num_episodes=args.num_episodes,
        save_path=f"{root_path}/output/struct_diffusion",
        num_save_digits=8,
        debug=True,
        seed=0,
    )
