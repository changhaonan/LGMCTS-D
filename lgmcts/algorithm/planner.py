"""Planner interface for LGMCTS
"""
from __future__ import annotations
import numpy as np
import cv2
import warnings
from lgmcts.algorithm.region_sampler import Region2DSampler, SampleData, SampleStatus
from lgmcts.algorithm.mcts import MCTS


class SamplingPlanner:
    """Sequential sampling, we provide a set of goals, and sample them one by one"""

    def __init__(self, sampler: Region2DSampler, **kwargs):
        self.sampler = sampler
        self.n_samples = kwargs.get("n_samples", 5)

    def reset(self):
        self.sampler.reset()

    def plan(self, goals: list[SampleData], algo: str, **kwargs):
        """Plan a sequence of goals
        Args:
            goals: list of SampleData
            algo: algorithm to use
        """
        if algo == "seq":
            return self.plan_seq(goals, **kwargs)
        elif algo == "mcts":
            return self.plan_mcts(goals, **kwargs)
        else:
            raise NotImplementedError

    def plan_seq(self, goals: list[SampleData], **kwargs):
        """Plan a sequence of goals
        Args:
            goals: list of SampleData
        """
        seed = kwargs.get("seed", 0)  # update seed
        self.sampler.rng = np.random.default_rng(seed=seed)
        prior_dict = kwargs.get("prior_dict", {})
        debug = kwargs.get("debug", False)
        sampled_obj_poses_pix = {}  # keep track of sampled object poses
        action_list = []
        cur_obj_poses = self.sampler.get_object_poses()
        for sample_data in goals:
            # the prior is where the joint sampling happens
            if sample_data.pattern in prior_dict:
                prior, pattern_info = prior_dict[sample_data.pattern].gen_prior(
                    self.sampler.grid_size, self.sampler.rng,
                    obj_id=sample_data.obj_id,
                    obj_ids=sample_data.obj_ids,
                    obj_poses_pix=sampled_obj_poses_pix,
                    sample_info=sample_data.sample_info,)
                pose_wd, pose_rg, sample_status, _ = self.sampler.sample(sample_data.obj_id, self.n_samples, prior, allow_outside=False)
                if sample_status is not SampleStatus.SUCCESS:
                    warnings.warn(f"Sample {sample_data.obj_id} failed")
                    continue
                # use the first one
                # mark this as sampled
                sampled_obj_poses_pix[sample_data.obj_id] = pose_rg[0, :2]
                # update the pose in sampler
                self.sampler.set_object_pose(sample_data.obj_id, pose_wd[0])
                # add to plan result
                action_list.append({
                    "obj_id": sample_data.obj_id,
                    "old_pose": cur_obj_poses[sample_data.obj_id].astype(np.float32),
                    "new_pose": pose_wd[0].astype(np.float32),
                })
                if debug:
                    print(f"old: {action_list[-1]['old_pose'][:3]}, new: {action_list[-1]['new_pose'][:3]}")
                    self.sampler.visualize()  # show the new pose
        return True, action_list

    def plan_mcts(self, goals: list[SampleData], **kwargs):
        """
        Task planning with MCTS
        Args:
            goals: list of SampleData
        Return:
            action_list: list of actions
        """
        seed = kwargs.get("seed", 0)  # update seed
        prior_dict = kwargs.get("prior_dict", {})
        reward_mode = kwargs.get('reward_mode', 'same')
        max_iter = kwargs.get("max_iter", 10000)
        is_virtual = kwargs.get("is_virtual", False)
        sampled_obj_poses_pix = {}  # keep track of sampled object poses
        action_list = []
        cur_obj_poses = self.sampler.get_object_poses()

        sampler_planner = MCTS(
            region_sampler=self.sampler,
            L=goals,
            obj_support_tree=self.sampler.obj_support_tree,
            prior_dict=prior_dict,
            n_samples=self.n_samples,
            reward_mode=reward_mode,
            is_virtual=is_virtual,
            verbose=True,
            seed=seed,
        )

        success = sampler_planner.search(max_iter=max_iter)
        return success, sampler_planner.action_list
