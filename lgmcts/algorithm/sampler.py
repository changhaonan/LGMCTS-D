"""Sampling interface, Currently we only consider 2.5D cases"""
from __future__ import annotations
import numpy as np
from typing import NamedTuple
from dataclasses import dataclass
from lgmcts.utils import misc_utils as utils
from lgmcts.utils import pybullet_utils as pb_utils
from lgmcts.algorithm.region_sampler import Region2DSampler

## Sample Data representation

@dataclass
class SampleData:
    """Sample representation"""
    pattern: str
    obj_id: int
    obj_ids: list[int]  # all included objects
    obj_poses: dict[int, np.ndarray] = {}  # poses that are already sampled

## Sampler examples

class Sampler:
    """Sampler class is function class"""
    @staticmethod
    def sample(self, env, sample_data: SampleData):
        """Sample"""
        raise NotImplementedError

class PairRelSampler(Sampler):
    @staticmethod
    def sample(self, env, sample_data: SampleData, obj_states: dict):
        """Sample using pairwise reltaionship"""
        ## Step 1: get pose
        
        raise NotImplementedError



if __name__ == "__main__":
    sampler = Sampler()
    sampler.reset()
    sampler.update({})
    sampler.sample_once(Pattern("random"), 0, [], {})
    sampler.sample_once(Pattern("circle"), 0, [1, 2, 3, 4], {})
