"""Planner interface for LGMCTS
"""
from __future__ import annotations
import numpy as np
import cv2
from lgmcts.algorithm.region_sampler import Region2DSampler, SampleData, SampleStatus


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
        else:
            raise NotImplementedError

    def plan_seq(self, goals: list[SampleData], **kwargs):
        """Plan a sequence of goals
        Args:
            goals: list of SampleData
        """
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
                    obj_poses_pix=sampled_obj_poses_pix)
                pose_wd, pose_rg, sample_status, _ = self.sampler.sample(sample_data.obj_id, self.n_samples, prior, allow_outside=False)
                if sample_status is not SampleStatus.SUCCESS:
                    print(f"Sample {sample_data.obj_id} failed")
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
                    cv2.imshow("prior", prior)  # show prior
                    cv2.waitKey(0)
                    self.sampler.visualize()  # show the new pose
        return action_list

