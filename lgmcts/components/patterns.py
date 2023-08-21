"""Define patterns here"""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import warnings
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
        """Generate line prior, based on if other objects are already sampled, 
            we will have different generate pattern.
        """
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 1)
        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        for id in obj_ids:
            if id != obj_id and id in obj_poses_pix:
                rel_obj_ids.append(id)
                rel_obj_poses_pix.append(obj_poses_pix[id])
        # if no other objects are sampled, we generate a random line
        height, width = img_size
        prior = np.zeros(img_size, dtype=np.float32)
        if len(rel_obj_ids) == 0:
            i0 = rng.integers(0, 4)  # select one of 4 borders
            i1 = (rng.integers(1, 4) + i0) % 4  # select one of 3 other borders
            i = [i0, i1]
            # select one point on each border
            x0 = rng.integers(0, width)
            y0 = rng.integers(0, height)
            x1 = rng.integers(0, width)
            y1 = rng.integers(0, height)
        elif len(rel_obj_ids) == 1:
            # random pixel
            x0 = rng.integers(0, width)
            y0 = rng.integers(0, height)
            x1 = rel_obj_poses_pix[0][0]
            y1 = rel_obj_poses_pix[0][1]
        else:
            # if more than one object is sampled, we generate a line based on the objects
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rel_obj_poses_pix[1][0]
            y1 = rel_obj_poses_pix[1][1]
            # compute the line that passing (xc1, yc1) & (xc2, yc2), both sides ending at borders
        ## Draw lines & extend to the borders
        # calculate the line's equation: y = mx + c
        if x1 - x0 == 0:  # vertical line
            start_point = (x0, 0)
            end_point = (x0, height-1)
            cv2.line(prior, (x0, 0), (x0, height-1), 1.0, thickness)
        elif y1 - y0 == 0:  # horizontal line
            start_point = (0, y0)
            end_point = (width-1, y0)
            cv2.line(prior, (0, y0), (width-1, y0), 1.0, thickness)
        else:
            m = (y1 - y0) / (x1 - x0)
            c = y0 - m * x0

            # Calculate intersection with the boundaries
            x_at_y0 = int(-c / (m+1e-6))
            x_at_y_max = int((height - c) / (m+1e-6))

            y_at_x0 = int(c)
            y_at_x_max = int(m * width + c)

            # Find points on the prior boundaries
            pt_candidate = [(x_at_y0, 0), (x_at_y_max, height-1), (0, y_at_x0), (width-1, y_at_x_max)]
            pt_candidate = [pt for pt in pt_candidate if 0 <= pt[0] <= width-1 and 0 <= pt[1] <= height-1]
            # sort by x
            pt_candidate = sorted(pt_candidate, key=lambda x: x[0])
            # Draw the line on the prior
            start_point = pt_candidate[0]
            end_point = pt_candidate[-1]
            cv2.line(prior, pt_candidate[0], pt_candidate[1], 1.0, thickness)

        # Debug
        # cv2.imshow("prior", prior)
        # cv2.waitKey(0)
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:line"
        pattern_info["position_pixel"] = [start_point[0], start_point[1], end_point[0], end_point[1]]
        pattern_info["rotation"] = [0, 0, np.arctan2(y1 - y0, x1 - x0)]
        
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if obj poses meets a line pattern"""
        assert "pattern_info" in kwargs, "Pattern info must be provided!"
        pattern_info = kwargs["pattern_info"]
        # check if p2l distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        if len(pattern_info["obj_ids"]) == 0:
            warnings.warn("No object in the pattern!")
            return False
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