"""Define patterns here"""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import cv2


class Pattern(ABC):
    """Base pattern type, needs to implement generate, check method
    """
    name: str = ""

    @abstractclassmethod
    def gen_prior(cls, size, rng, **kwargs):
        """Generate a pattern prior:
        Args: 
            rng: random generator
        """
        raise NotImplementedError
    
    @abstractclassmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if the object states meet the pattern requirement
        """
        raise NotImplementedError


## Implementation of patterns

class LinePattern(Pattern):
    """Line pattern, obj poses should formulate a line"""
    name = "line"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate line prior"""
        height, width = img_size
        prior = np.zeros(img_size, dtype=np.float32)
        i0 = rng.integers(0, 4)  # select one of 4 borders
        i1 = (rng.integers(1, 4) + i0) % 4  # select one of 3 other borders
        i = [i0, i1]
        # select one point on each border
        x0 = rng.integers(0, width) if i[0] % 2 == 0 else (1 - i[0]//2) * width - 1
        y0 = rng.integers(0, height) if i[0] % 2 == 1 else (1 - i[0]//2) * height - 1
        x1 = rng.integers(0, width) if i[1] % 2 == 0 else (1 - i[1]//2) * width - 1
        y1 = rng.integers(0, height) if i[1] % 2 == 1 else (1 - i[1]//2) * height - 1

        # Draw the line on the image
        cv2.line(prior, (x0, y0), (x1, y1), 1.0, 1)
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:line"
        pattern_info["position_pixel"] = [int(x0), int(y0)]
        pattern_info["rotation"] = [0, 0, np.arctan2(y1 - y0, x1 - x0)]
        
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if obj poses meets a line pattern"""
        assert "pattern_info" in kwargs, "Pattern info must be provided!"
        pattern_info = kwargs["pattern_info"]
        # check if p2l distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        # assemble obj_poses
        obj_poses_pattern = []
        for obj_id in pattern_info["obj_ids"]:
            obj_poses_pattern.append(obj_poses[obj_id][:3])
        obj_poses_pattern = np.vstack(obj_poses_pattern)
        # get the up most and low most points first"""
        lo_idx = np.argmax(obj_poses_pattern[:, 1], axis=-1)
        hi_idx = np.argmin(obj_poses_pattern[:, 1], axis=-1)
        lo_pose = obj_poses_pattern[lo_idx, :2]
        hi_pose = obj_poses_pattern[hi_idx, :2]
        k = (hi_pose - lo_pose) / np.linalg.norm(hi_pose - lo_pose)
        o = hi_pose
        # 
        dists = cls.dist_p2l(obj_poses_pattern[:, :2], o[None, :], k[None, :])
        return not(np.max(dists) > threshold)

    @classmethod
    def dist_p2l(cls, p, o, k):
        """(Vectorized meethod) disance, point to line"""
        op = p - o
        k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
        op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
        op_ver = op - op_proj
        return np.linalg.norm(op_ver, axis=-1)

#TODO: we need to add more patterns here, e.g. circle, rectangle, spatial, etc. @Alex


## PATTERN DICT

PATTERN_DICT = {
    "line": LinePattern
}