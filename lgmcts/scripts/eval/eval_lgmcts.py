"""Evaluate the performace of lgmcts system"""
from __future__ import annotations
import os
import lgmcts
import numpy as np
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as misc_utils
import lgmcts.utils.pybullet_utils as pb_utils
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.algorithm import Sampler
##
from lgmcts.algorithm.region_sampler import Region2DSampler
from lgmcts.components.patterns import PATTERN_DICT

## Utils method

def separate_pcd_pose(obj_ids, pcd_batch, pose_batch, max_pcd_size):
    """Seperate point cloud from tensor structure
    Args:
        obj_ids: list of object ids
        pcd_batch: tensor of shape (batch_size, max_pcd_size, 3)
    """
    pcd_batch = pcd_batch.reshape([-1, max_pcd_size, 3])
    pose_batch = pose_batch.reshape([-1, 7])
    pcd_list = []
    pose_list = []
    for i, obj_id in enumerate(obj_ids):
        pcd = pcd_batch[i]
        pose = pose_batch[i]
        # remove 0 padding from pcd
        pcd = pcd[pcd[:, 0] != 0]
        pcd_list.append(pcd)
        pose_list.append(pose)
    return obj_ids, pcd_list, pose_list


def build_env_and_task(
    task_name: str,
    task_kwargs: dict | None,
    modalities,
    seed: int | None = None,
    debug: bool = False,
):
    env = lgmcts.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed, debug=debug, display_debug_window=debug,
    )
    task = env.task
    return env, task


## Eval method

def eval_online(seed: int = 0):
    """Eval from newly generated scene"""
    task_name = "struct_rearrange"
    ## TODO: load constants from config
    resolution = 0.01
    num_epochs = 10
    ## Init
    env, task = build_env_and_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=seed,
        debug=True,
    )
    bounds = env.bounds  # (3, 2)
    grid_size = (int((env.bounds[0, 1] - env.bounds[0, 0]) / resolution), 
        int((env.bounds[1, 1] - env.bounds[1, 0]) / resolution))
    world2region = np.eye(4, dtype=np.float32)
    world2region[:3, 3] = -bounds[:, 0]
    region_sampler = Region2DSampler(resolution, grid_size, world2region=world2region)
    prompt_generator = PromptGenerator(env.rng)
    result_list = []

    for i in range(num_epochs):
        obs = env.reset()
        region_sampler.reset()
        prompt_generator.reset()
        # generate goal
        prompt_str, obs = task.gen_goal_config(env, prompt_generator)
        obs = task.gen_start_config(env)
        
        # Step 1: get observation
        obj_pcds = obs["point_cloud"]["top"]
        obj_poses = obs["poses"]["top"]
        obj_lists = env.obj_ids["rigid"]
        obj_names = [env.obj_id_reverse_mapping[obj_id]["obj_name"] for obj_id in obj_lists]
        max_pcd_size = env.obs_img_size[0] * env.obs_img_size[1]
        obj_lists, obj_pcd_list, obj_pose_list = separate_pcd_pose(obj_lists, obj_pcds, obj_poses, max_pcd_size)
        
        # Step 2: init region sampler
        for j, (obj_id, obj_name, obj_pcd, obj_pose) in enumerate(zip(obj_lists, obj_names, obj_pcd_list, obj_pose_list)):
            # compute the pos_ref
            obj_pcd_center = obj_pcd.mean(axis=0)
            obj_pcd -= obj_pcd_center
            # pos_ref = obj_pose[:3] - obj_pcd_center
            pos_ref = None
            color = np.random.rand(3) * 255
            # DEBUG
            # misc_utils.plot_3d("test", obj_pcd, "red")
            # add object to region sampler
            region_sampler.add_object(
                obj_id=obj_id,
                points=obj_pcd, 
                pos_ref=pos_ref,
                name=obj_name,
                color=color
            )
            # set object pose
            region_sampler.set_object_pose(obj_id, obj_pose[:3])
            # get point cloud
        
            # region_sampler.visualize()
            # region_sampler.visualize_3d()

        # Step 3: Exectue actions
        # parse the sample strategy from prompt
        sample_goals = task.goals  # TODO: This is temp cheat by directly loading the goal @Haonan
        # run mcts to get plan
        # TODO: currently using a simple sequential sampler @Haonan
        n_samples = 10
        sample_goal = sample_goals[0]
        pattern_type = sample_goal["type"].split(":")[-1]
        pattern_prior = PATTERN_DICT[pattern_type].gen_prior(env.ws_map_size, env.rng)
        # sample_poses = region_sample.sample()

        # execute action

        # Step 4: Collect result
        obj_poses = env.get_obj_poses()
        result = task.check_success(obj_poses=obj_poses)
        if result.success:
            result_list.append(1)
        else:
            result_list.append(0)
        # Step 5: Log
        print(f"Epoch {i}: {result.success}")

    ## Analysis the result
    print(f"Success rate: {np.mean(result_list)}")

if __name__ == "__main__":
    eval_online()