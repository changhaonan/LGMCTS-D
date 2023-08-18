from __future__ import annotations
import os
import numpy as np
import lgmcts
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
import lgmcts.utils.misc_utils as utils
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.algorithm.sampler import Sampler, SampleData
from lgmcts.algorithm.region_sampler import Region2DSampler
from lgmcts.components.patterns import PATTERN_DICT


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


if __name__ == '__main__':
    task_name = "struct_rearrange"
    resolution = 0.01
    n_samples = 1

    env, task = build_env_and_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=True,
    )
    bounds = env.bounds  # (3, 2)
    grid_size = (int((env.bounds[0, 1] - env.bounds[0, 0]) / resolution), 
        int((env.bounds[1, 1] - env.bounds[1, 0]) / resolution))
    world2region = np.eye(4, dtype=np.float32)
    world2region[:3, 3] = -bounds[:, 0]
    region_sampler = Region2DSampler(resolution, grid_size, world2region=world2region)
    prompt_generator = PromptGenerator(env.rng)

    for i in range(10):
        # reset
        obs = env.reset()
        prompt_generator.reset()
        region_sampler.reset()
        # generate goal
        prompt_str, obs = task.gen_goal_config(env, prompt_generator)
        obs = task.gen_start_config(env)
        
        # Step 1: get observation
        obj_pcds = obs["point_cloud"]["top"]
        obj_poses = obs["poses"]["top"]
        obj_lists = env.obj_ids["rigid"]
        obj_names = [env.obj_id_reverse_mapping[obj_id]["obj_name"] for obj_id in obj_lists]
        max_pcd_size = env.obs_img_size[0] * env.obs_img_size[1]
        obj_lists, obj_pcd_list, obj_pose_list = utils.separate_pcd_pose(obj_lists, obj_pcds, obj_poses, max_pcd_size)
        
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
        region_sampler.visualize()
        ## Step 3. build a sampler based on the goal (from goal is cheat, we want to from LLM in the future)
        goals = task.goals
        L = []
        for j, goal in enumerate(goals):
            goal_obj_ids = goal["obj_ids"]
            for goal_obj_id in goal_obj_ids:
                sample_data = SampleData(goal["type"].split(":")[-1], goal_obj_id, goal["obj_ids"], {})
                L.append(sample_data)

        ## Step 4. sample a goal
        sampled_obj_poses_pix = {}
        for sample_data in L:
            # the prior is where the joint sampling happens
            if sample_data.pattern in PATTERN_DICT:
                prior, pattern_info = PATTERN_DICT[sample_data.pattern].gen_prior(
                    grid_size, env.rng, 
                    obj_id=sample_data.obj_id, 
                    obj_ids=sample_data.obj_ids,
                    obj_poses_pix=sampled_obj_poses_pix)
                pose_wd, pose_rg, sample_status, _ = region_sampler.sample(sample_data.obj_id, n_samples, prior)
                # mark this as sampled
                sampled_obj_poses_pix[sample_data.obj_id] = pose_rg[0, :2]
                # update the pose in sampler
                region_sampler.set_object_pose(sample_data.obj_id, pose_wd[0])

        print(f"==== Episode {i} ====")
        print(prompt_str)