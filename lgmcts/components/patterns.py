"""Define patterns here"""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import math
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import minimize

import cv2

PATTERN_CONSTANTS = {
    "line": {
        "line_len": {
            "L": [0.2, 0.2],
            "M": [0.1, 0.1],
            "S": [0.05, 0.05]
        }
    },
    "circle": {
        "radius": {
            "L": [0.4, 0.5],
            "M": [0.1, 0.4],
            "S": [0.1, 0.2]
        }
    },
    "rectangle": {
        "edge_len": {
            "L": [0.45, 0.5],
            "M": [0.35, 0.8],
            "S": [0.3, 0.35]
        }
    }
}


class Pattern(ABC):
    """Base pattern type, needs to implement generate, check method
    """
    name: str = ""
    _num_limit = [0, 100]  # [min, max]

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate a pattern prior:
        Args: 
            rng: random generator
        """
        height, width = img_size[0], img_size[1]
        prior = np.ones([height, width], dtype=np.float32)
        pattern_info = {}
        pattern_info["type"] = "pattern:uniform"
        pattern_info["angle"] = 0.0
        pattern_info["disable_collision_check"] = False
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if the object states meet the pattern requirement
        """
        return True


class RigidPattern(Pattern):
    """Rigid pattern"""
    name = "rigid"
    _num_limit = [1, 1]

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate a pattern prior:
        Args: 
            rng: random generator
        """
        sample_info = kwargs["sample_info"]
        obj_id = kwargs["obj_id"]
        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)

        # FIXME: currently only position is implemented
        pos_pix = sample_info["pos_pix"][obj_id]
        x0, y0 = pos_pix[1], pos_pix[0]
        cv2.circle(prior, (x0, y0), 1, 1.0, -1)

        pattern_info = {}
        pattern_info["type"] = "pattern:rigid"
        pattern_info["angle"] = 0.0
        pattern_info["disable_collision_check"] = False
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if the object states meet the pattern requirement
        """
        return True

# Implementation of patterns


class LinePattern(Pattern):
    """Line pattern, obj poses should formulate a line"""
    name = "line"
    _num_limit = [0, 10]

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate line prior, based on if other objects are already sampled, 
            we will have different generate pattern.
        """
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 1)
        try_center = kwargs.get("try_center", False)
        assert len(obj_ids) == 0 or (len(obj_ids) >= cls._num_limit[0] and len(obj_ids)
                                     <= cls._num_limit[1]), "Number of objects should be within the limit!"

        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        for id in obj_ids:
            if id != obj_id and id in obj_poses_pix:
                rel_obj_ids.append(id)
                rel_obj_poses_pix.append(obj_poses_pix[id])
        # if no other objects are sampled, we generate a random line
        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)

        # some constants
        scale_max = PATTERN_CONSTANTS["line"]["line_len"]["M"][0]
        scale_min = PATTERN_CONSTANTS["line"]["line_len"]["M"][1]
        scale = rng.random() * (scale_max - scale_min) + scale_min
        enable_vis = False
        block_vis = False

        if len(rel_obj_ids) == 0:
            if len(obj_ids) == 0:
                # pure pattern
                if try_center:
                    x0 = width // 2
                    y0 = height // 2
                else:
                    x0 = rng.integers(0, width)
                    y0 = rng.integers(0, height)
                if rng.random() > 0.5:
                    # horizontal line
                    cv2.line(prior, (0, y0), (x0 + width, y0), 1.0, thickness)
                    angle = 0.0
                else:
                    # vertical line
                    cv2.line(prior, (x0, 0), (x0, height), 1.0, thickness)
                    angle = np.pi / 2.0
            else:
                x0 = 0
                y0 = 0
                # no points are provided
                prior[:, :] = 1.0
                angle = rng.integers(0, 1) * np.pi / 2.0  # randomly select a horizontal or vertical line
                block_vis = True
        elif len(rel_obj_ids) == 1:
            # given one pix
            x0 = rel_obj_poses_pix[0][1]
            y0 = rel_obj_poses_pix[0][0]
            # random pixel
            if rng.random() > 0.5:
                # horizontal line
                x1, y1 = width, y0
            else:
                # vertical line
                x1, y1 = x0, height
            # HACK: create a line that is longest in the region
            # dist2corner = np.linalg.norm(np.array([x0, y0]) - np.array([[0, 0], [width, 0], [0, height], [width, height]]), axis=-1)
            # max_idx = np.argmax(dist2corner)
            # if max_idx == 0:
            #     x1, y1 = 0, 0
            # elif max_idx == 1:
            #     x1, y1 = width, 0
            # elif max_idx == 2:
            #     x1, y1 = 0, height
            # else:
            #     x1, y1 = width, height
            cls.draw_line(prior, x0, y0, x1, y1, thickness)
            # angle = math.atan2(y1 - y0, x1 - x0) + np.pi / 2.0
            angle = math.atan2(y1 - y0, x1 - x0)
        else:
            rel_obj_poses_pix = np.vstack(rel_obj_poses_pix)
            # if more than one object is sampled, we generate a line based on the objects
            dist_mat = np.linalg.norm(rel_obj_poses_pix[:, :2] - rel_obj_poses_pix[:, None, :2], axis=-1)
            max_idx = np.argmax(dist_mat)
            max_idx = np.unravel_index(max_idx, dist_mat.shape)
            lo_idx, hi_idx = max_idx[0], max_idx[1]  # the two furthest points
            x0 = rel_obj_poses_pix[lo_idx][1]
            y0 = rel_obj_poses_pix[lo_idx][0]
            x1 = rel_obj_poses_pix[hi_idx][1]
            y1 = rel_obj_poses_pix[hi_idx][0]
            angle = math.atan2(y1 - y0, x1 - x0)
            cls.draw_line(prior, x0, y0, x1, y1, thickness)

        # Debug
        if enable_vis and not block_vis:
            vis_prior = prior.copy()
            for id, rel_pix in zip(rel_obj_ids, rel_obj_poses_pix):
                cv2.circle(vis_prior, (rel_pix[1], rel_pix[0]), 5, 1.0, 1)
                cv2.putText(vis_prior, f"{id}", (rel_pix[1] - 30, rel_pix[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 0.8, 1)
            # put text
            cv2.putText(vis_prior, f"angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0.8, 1)
            cv2.imshow("prior", vis_prior)
            cv2.waitKey(1)
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:line"
        pattern_info["angle"] = angle
        pattern_info["min_length"] = scale_max
        pattern_info["max_length"] = scale_min
        pattern_info["length"] = scale
        pattern_info["position_pixel"] = [int(x0)/width, float(int(y0)/height), 0.0]
        pattern_info["rotation"] = [0.0, 0.0, angle]
        pattern_info["disable_collision_check"] = False
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray] | None = None, obj_poses_pattern: np.ndarray | None = None, **kwargs):
        """Check if obj poses meets a line pattern"""
        pattern_info = kwargs["pattern_info"]
        # check if p2l distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        # assemble obj_poses
        if obj_poses_pattern is None:
            obj_poses_pattern = []
            for obj_id in pattern_info["obj_ids"]:
                obj_poses_pattern.append(obj_poses[obj_id][:3])
            obj_poses_pattern = np.vstack(obj_poses_pattern)

        # get the two furthest points
        dist_mat = np.linalg.norm(obj_poses_pattern[:, :2] - obj_poses_pattern[:, None, :2], axis=-1)
        max_idx = np.argmax(dist_mat)
        max_idx = np.unravel_index(max_idx, dist_mat.shape)

        lo_idx, hi_idx = max_idx[0], max_idx[1]
        lo_pose = obj_poses_pattern[lo_idx, :2]
        hi_pose = obj_poses_pattern[hi_idx, :2]
        k = (hi_pose - lo_pose) / np.linalg.norm(hi_pose - lo_pose)
        o = hi_pose
        #
        dists = cls.dist_p2l(obj_poses_pattern[:, :2], o[None, :], k[None, :])
        status = not (np.max(dists) > threshold)
        if not status:
            print(f"Line pattern check failed!, dist: {dists}")
        return status

    @classmethod
    def draw_line(cls, img, x0, y0, x1, y1, thickness, use_guassian=True):
        """Draw lines & extend to the borders"""
        height, width = img.shape[0], img.shape[1]
        # calculate the line's equation: y = mx + c
        if x1 - x0 == 0:  # vertical line
            start_point = (int(x0), 0)
            end_point = (int(x0), height-1)
        elif y1 - y0 == 0:  # horizontal line
            start_point = (0, int(y0))
            end_point = (width-1, int(y0))
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
        if use_guassian:
            cv2.line(img, start_point, end_point, 1.0, 3 * thickness)
            cv2.line(img, start_point, end_point, 2.0, thickness)
        else:
            cv2.line(img, start_point, end_point, 1.0, thickness)

    @classmethod
    def dist_p2l(cls, p, o, k):
        """(Vectorized meethod) disance, point to line"""
        op = p - o
        k = np.repeat(k, [op.shape[0]]).reshape([2, -1]).T
        op_proj = np.sum(np.multiply(op, k), axis=-1)[..., None] * k
        op_ver = op - op_proj
        return np.linalg.norm(op_ver, axis=-1)


class CirclePattern(Pattern):
    """Circle pattern, obj poses should formulate a circle"""
    name = "circle"
    _num_limit = [2, 100]  # at least 2 points

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate circle prior"""
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 3)
        rel_size = kwargs.get("rel_size", "M")
        try_center = kwargs.get("try_center", False)
        assert len(obj_ids) == 0 or (len(obj_ids) >= cls._num_limit[0] and len(obj_ids)
                                     <= cls._num_limit[1]), "Number of objects should be within the limit!"

        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        for id in obj_ids:
            if id != obj_id and id in obj_poses_pix:
                rel_obj_ids.append(id)
                rel_obj_poses_pix.append(obj_poses_pix[id])
        assert len(rel_obj_ids) == len(rel_obj_poses_pix), "Number of relative objects should be the same!"

        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)

        # some constants
        scale_max = PATTERN_CONSTANTS["circle"]["radius"][rel_size][0]
        scale_min = PATTERN_CONSTANTS["circle"]["radius"][rel_size][1]
        scale = rng.random() * (scale_max - scale_min) + scale_min
        radius = int(scale * (min(height, width)))
        segments = len(obj_ids) if len(obj_ids) % 2 == 0 else len(obj_ids) + 1
        if segments <= 2:
            segments = 8  # default

        block_vis = False
        enable_vis = False
        if len(rel_obj_ids) == 0:
            if len(obj_ids) == 0:
                # FIXME: Currently, this doesn't support generate proper angle
                # pure pattern
                if try_center:
                    center_x = width // 2
                    center_y = height // 2
                else:
                    center_x = rng.integers(radius, width - radius)
                    center_y = rng.integers(radius, height - radius)
                cls.draw_seg_circle(prior, (center_x, center_y), radius, 1.0, thickness, segments)
                angle = 0.0
            else:
                # no points are provided
                prior[radius:height - radius, radius:width - radius] = 1.0
                angle = -np.pi / 2.0
                block_vis = True
        elif len(rel_obj_ids) == 1:
            # given an pix, the next point is on the other side of circle
            # HACK: make sure the circle is within the region
            angle = np.pi / 2.0
            x0, y0 = rel_obj_poses_pix[0][1], rel_obj_poses_pix[0][0]
            if cls.check_circle_in_region((x0 - radius, y0), radius, height, width):
                cv2.circle(prior, (x0 - 2 * radius, y0), 1, 1.0, thickness)
            if cls.check_circle_in_region((x0 + radius, y0), radius, height, width):
                cv2.circle(prior, (x0 + 2 * radius, y0), 1, 1.0, thickness)
            center_xs = [x0 - radius, x0 + radius]
            center_ys = [y0, y0]
        else:
            # if more than one object is sampled, we generate a circle based on the objects
            rel_obj_poses_pix = [pix[:2] for pix in rel_obj_poses_pix]
            points = np.array(rel_obj_poses_pix)
            points = points[:, [1, 0]]  # swap x, y
            if len(rel_obj_ids) == 2:
                # Find the minimum enclosing circle of first 3 points
                center_x = (points[0, 0] + points[1, 0]) / 2.0
                center_y = (points[0, 1] + points[1, 1]) / 2.0
                radius = np.linalg.norm(points[0, :] - points[1, :]) / 2.0
            else:
                # Find the minimum enclosing circle of first 3 points
                try:
                    points = points[:3, :]
                except:
                    print(points)
                center, radius = cls.cercle_circonscrit(points)
                center_x, center_y = center[0], center[1]
            # provides two candidates
            angle_circle = (2.0 * np.pi / segments) * (len(rel_obj_ids) // 2)
            cv2.circle(prior, (int(center_x + radius * math.cos(angle_circle)),
                       int(center_y + radius * math.sin(angle_circle))), thickness, 1.0, -1)
            cv2.circle(prior, (int(center_x - radius * math.cos(angle_circle)),
                       int(center_y - radius * math.sin(angle_circle))), thickness, 1.0, -1)
            angle = - np.pi / 2.0 + angle_circle
            center_xs = [center_x]
            center_ys = [center_y]
        if not block_vis and enable_vis:
            vis_prior = prior.copy()
            for (center_x, center_y) in zip(center_xs, center_ys):
                cv2.circle(vis_prior, (int(center_x), int(center_y)), int(radius), 1.0, 1)
            # add existing
            for id, rel_pix in zip(rel_obj_ids, rel_obj_poses_pix):
                cv2.circle(vis_prior, (rel_pix[1], rel_pix[0]), 5, 1.0, 1)
                cv2.putText(vis_prior, f"{id}", (rel_pix[1] - 30, rel_pix[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, 0.8, 1)
            # put text
            cv2.putText(vis_prior, f"angle: {angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0.8, 1)
            cv2.imshow("cricle", vis_prior)
            cv2.waitKey(1)
            # pass
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:circle"
        pattern_info["obj_ids"] = obj_ids
        pattern_info["angle"] = angle
        pattern_info["disable_collision_check"] = False
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray] | None = None, obj_poses_pattern: np.ndarray | None = None, **kwargs):
        """Check if obj poses meet a circle pattern"""
        pattern_info = kwargs["pattern_info"]

        # Check if p2c distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        # assemble obj_poses
        if obj_poses_pattern is None:
            obj_poses_pattern = []
            for obj_id in pattern_info["obj_ids"]:
                obj_poses_pattern.append(obj_poses[obj_id][:3])
            obj_poses_pattern = np.vstack(obj_poses_pattern)

        if len(obj_poses_pattern) <= 3:
            return True
        else:
            # Calculate distances from object poses to the circle's center
            dists = cls.dist_rad(np.array([[sublist[0], sublist[1]] for sublist in obj_poses_pattern]))
            # print(dists)

            return not (np.max(dists) > threshold)

    @classmethod
    def dist_rad(cls, point_list):
        def circle_equation(params, points):
            cx, cy, r = params
            x_vals = [sub_array[0] for sub_array in points]
            y_vals = [sub_array[1] for sub_array in points]
            distances = np.sqrt((x_vals - cx)**2 + (y_vals - cy)**2)
            return distances - r

        # Function to optimize
        def objective(params, *args):
            points = args[0]
            return np.sum(circle_equation(params, points)**2)

        initial_guess = [0, 0, 1]
        constraints = ({'type': 'ineq', 'fun': lambda x: x[2]})
        result = minimize(objective, initial_guess, args=(point_list,), method='trust-constr', constraints=constraints)
        distances = circle_equation(result.x, point_list)
        return distances

    @classmethod
    def check_circle_in_region(cls, center, radius, height, width):
        """Check if the circle is within the region"""
        if center[0] - radius < 0 or center[0] + radius > width:
            return False
        if center[1] - radius < 0 or center[1] + radius > height:
            return False
        return True

    @classmethod
    def cercle_circonscrit(cls, T):
        (x1, y1), (x2, y2), (x3, y3) = T
        A = np.array([[x3 - x1, y3 - y1], [x3 - x2, y3 - y2]])
        Y = np.array([(x3**2 + y3**2 - x1**2 - y1**2), (x3**2+y3**2 - x2**2-y2**2)])
        if np.linalg.det(A) == 0:
            return False
        Ainv = np.linalg.inv(A)
        X = 0.5 * np.dot(Ainv, Y)
        x, y = X[0], X[1]
        r = np.sqrt((x - x1)**2 + (y - y1)**2)
        return (x, y), r

    @classmethod
    def draw_seg_circle(cls, img, center, radius, color, thickness, segments: int = 6):
        """Draw a circle segment on the image"""
        # compute the angle between two segments
        angle = 2 * math.pi / segments
        # compute the points on the circle
        for i in range(segments):
            x = radius * math.cos(i * angle) + center[0]
            y = radius * math.sin(i * angle) + center[1]
            cv2.circle(img, (int(x), int(y)), thickness, color, -1)


class TowerPattern(Pattern):
    """Tower pattern, obj poses should formulate a tower"""
    name = "tower"
    _num_limit = [2, 100]  # at least 3 points

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 3)
        rel_size = kwargs.get("rel_size", "M")
        assert len(obj_ids) == 0 or (len(obj_ids) >= cls._num_limit[0] and len(obj_ids)
                                     <= cls._num_limit[1]), "Number of objects should be within the limit!"

        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        for id in obj_ids:
            if id != obj_id and id in obj_poses_pix:
                rel_obj_ids.append(id)
                rel_obj_poses_pix.append(obj_poses_pix[id])

        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)
        if len(rel_obj_ids) == 0:
            # no points are provided
            prior[:, :] = 1.0
            angle = 0.0
        else:
            # provided points
            rel_obj_poses_pix = [pix[:2] for pix in rel_obj_poses_pix]
            points = np.array(rel_obj_poses_pix)
            prior[points[:, 0], points[:, 1]] = 1.0  # stack on previous objects
            angle = 0.0
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:tower"
        pattern_info["obj_ids"] = obj_ids
        pattern_info["angle"] = angle
        pattern_info["disable_collision_check"] = True  # disable collision check
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray] | None = None, obj_poses_pattern: np.ndarray | None = None, **kwargs):
        """Check if obj poses meet a circle pattern"""
        pattern_info = kwargs["pattern_info"]

        # Check if p2c distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        # assemble obj_poses
        if obj_poses_pattern is None:
            obj_poses_pattern = []
            for obj_id in pattern_info["obj_ids"]:
                obj_poses_pattern.append(obj_poses[obj_id][:3])
            obj_poses_pattern = np.vstack(obj_poses_pattern)

        dist = np.linalg.norm(obj_poses_pattern[:, :2] - obj_poses_pattern[0, :2]) / obj_poses_pattern.shape[0]
        return not (dist > threshold)


class RectanglePattern(Pattern):
    """Rectangle pattern, obj poses should formulate a rectangle"""
    name = "rectangle"
    _num_limit = [3, 4]  # at least 3 points, at most 4 points

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate rectangle prior"""
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 1)
        rel_size = kwargs.get("rel_size", "M")
        assert len(obj_ids) == 0 or (len(obj_ids) >= cls._num_limit[0] and len(obj_ids)
                                     <= cls._num_limit[1]), "Number of objects should be within the limit!"

        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        for id in obj_ids:
            if id != obj_id and id in obj_poses_pix:
                rel_obj_ids.append(id)
                rel_obj_poses_pix.append(obj_poses_pix[id])

        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)

        # some constants
        clearance = int(0.1 * min(height, width))
        scale_max = PATTERN_CONSTANTS["rectangle"]["edge_len"][rel_size][0]
        scale_min = PATTERN_CONSTANTS["rectangle"]["edge_len"][rel_size][1]
        scale = rng.random() * (scale_max - scale_min) + scale_min
        edge_len = int(scale * (min(height, width) - clearance))
        if len(rel_obj_ids) == 0:
            if len(obj_ids) == 0:
                # pure pattern
                x0, y0, x1, y1, x2, y2, x3, y3 = cls.rec_in_region(edge_len, height, width, clearance)
                rect_points = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
                cv2.circle(prior, (x0, y0), 1, 1.0, -1)
                cv2.circle(prior, (x1, y1), 1, 1.0, -1)
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)
                cv2.circle(prior, (x3, y3), 1, 1.0, -1)
            else:
                # no points are provided
                prior[clearance:height-clearance, clearance:width-clearance] = 1.0
                rect_points = []
        elif len(rel_obj_ids) == 1:
            # if one point is already given, the next point should be on two lines, x = x0 and y = y0
            x0 = rel_obj_poses_pix[0][1]
            y0 = rel_obj_poses_pix[0][0]
            rect_points = [[x0, y0]]
            cv2.circle(prior, (x0 - edge_len, y0), 1, 1.0, -1)
            cv2.circle(prior, (x0 + edge_len, y0), 1, 1.0, -1)
            cv2.circle(prior, (x0, y0 - edge_len), 1, 1.0, -1)
            cv2.circle(prior, (x0, y0 + edge_len), 1, 1.0, -1)
        elif len(rel_obj_ids) == 2:
            # if two points are already given, the third point shuold formulate a right triangle with the two points
            x0 = rel_obj_poses_pix[0][1]
            y0 = rel_obj_poses_pix[0][0]
            x1 = rel_obj_poses_pix[1][1]
            y1 = rel_obj_poses_pix[1][0]
            if np.abs(x1 - x0) < np.abs(y1 - y0):
                # p3 should be on the same x axis as p1 or p2
                edge_len = np.abs(y1 - y0)
                x2 = x0 + edge_len
                y2 = y0
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)  # 1
                y2 = y1
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)  # 2
                x2 = x0 - edge_len
                y2 = y0
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)  # 3
                y2 = y1
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)  # 4
            else:
                # p3 should be on the same y axis as p1 or p2
                edge_len = np.abs(x1 - x0)
                y2 = y0 + edge_len
                x2 = x0
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)
                x2 = x1
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)
                y2 = y0 - edge_len
                x2 = x0
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)
                x2 = x1
                cv2.circle(prior, (x2, y2), 1, 1.0, -1)
            rect_points = [[x0, y0], [x1, y1], [x2, y2]]
        elif len(rel_obj_ids) == 3:
            # if three points are already given, the fourth one is fixed.
            x0 = rel_obj_poses_pix[0][1]
            y0 = rel_obj_poses_pix[0][0]
            x1 = rel_obj_poses_pix[1][1]
            y1 = rel_obj_poses_pix[1][0]
            x2 = rel_obj_poses_pix[2][1]
            y2 = rel_obj_poses_pix[2][0]
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            x3, y3 = cls.find_fourth_point(p0, p1, p2)
            cv2.circle(prior, (x3, y3), 1, 1.0, -1)
            rect_points = [p0, p1, p2]
        else:
            raise ValueError("Too many points are given!")

        # DEBUG
        debug_image = prior.copy()
        for i in range(len(rect_points)):
            cv2.circle(debug_image, (rect_points[i][0], rect_points[i][1]), 1, 1.0, -1)
        cv2.imshow("rectangle", debug_image)
        cv2.waitKey(1)
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:rectangle"
        pattern_info["corners"] = rect_points
        pattern_info["obj_ids"] = obj_ids
        pattern_info["disable_collision_check"] = False
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if obj poses meet a rectangle pattern"""
        assert "pattern_info" in kwargs, "Pattern info must be provided!"
        pattern_info = kwargs["pattern_info"]
        threshold = pattern_info.get("threshold", 0.1)

        # assemble obj_poses
        obj_poses_pattern = []
        for obj_id in pattern_info["obj_ids"]:
            obj_poses_pattern.append(obj_poses[obj_id][:2])
        obj_poses_pattern = np.vstack(obj_poses_pattern)
        if len(obj_poses_pattern) < 4:
            warnings.warn("not enough points")
            return False
        else:
            dists = cls.rec_dist(obj_poses_pattern)
            status = not (np.max(dists) > threshold)
            if not status:
                print(f"Rectangle pattern not met; dists: {dists}")
            return status

    @classmethod
    def rec_in_region(cls, edge_len, height, width, clearance):
        x0 = np.random.randint(clearance, width - edge_len - clearance)
        y0 = np.random.randint(clearance, height - edge_len - clearance)
        x1 = int(x0 + edge_len)
        y1 = y0
        x2 = x1
        y2 = int(y1 + edge_len)
        x3 = x0
        y3 = y2
        return x0, y0, x1, y1, x2, y2, x3, y3

    @classmethod
    def rec_dist(cls, rec_points):
        """Dist shift compared with same rec"""
        p0, p1, p2, p3 = rec_points
        dist_01 = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        dist_02 = math.sqrt((p0[0] - p2[0])**2 + (p0[1] - p2[1])**2)
        dist_03 = math.sqrt((p0[0] - p3[0])**2 + (p0[1] - p3[1])**2)
        dist_12 = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        dist_13 = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
        dist_23 = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
        edge_dists = [dist_01, dist_02, dist_03, dist_12, dist_13, dist_23]
        edge_dists.sort()
        edge_dists = np.array(edge_dists[:4])
        dist_variance = (edge_dists - edge_dists.mean()).max()
        return dist_variance

    @classmethod
    def find_fourth_point(cls, p1, p2, p3):
        # HACK: only support axia aligned rec
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        # boundary
        x_min = np.min([p1[0], p2[0], p3[0]])
        x_max = np.max([p1[0], p2[0], p3[0]])
        y_min = np.min([p1[1], p2[1], p3[1]])
        y_max = np.max([p1[1], p2[1], p3[1]])
        ps_cand = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])  # (4, 2)
        # dist between candidate and ps
        ps = np.vstack([p1, p2, p3])  # (3, 2)
        #
        dist_mat = np.zeros([4, 3], dtype=np.float32)
        for i in range(4):
            for j in range(3):
                dist_mat[i, j] = np.linalg.norm(ps[j] - ps_cand[i])
        dist_mat = dist_mat.min(axis=1)  # (4, 1)
        id_max = np.argmax(dist_mat)  # id of max distance from candidates
        return ps_cand[id_max]

    @classmethod
    def find_diagonal(cls, p1, p2, p3):
        for rotation in range(3):
            if abs(cls.calculate_angle(p1, p2, p3) - 90) < 1:
                return p2, p1, p3
            else:
                temp = p1
                p1 = p2
                p2 = p3
                p3 = temp
        return p1, p2, p3

    @classmethod
    def calculate_angle(cls, p1, p2, p3):
        vector1 = np.array(p1) - np.array(p2)
        vector3 = np.array(p3) - np.array(p2)

        dot_product = np.dot(vector1, vector3)
        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector3)

        angle_radians = np.arccos(dot_product / magnitude_product)
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    @classmethod
    def dist_corners(cls, point_list):
        def objective_function(params, points):
            x_center, y_center, width, height, angle = params

            corner_points = np.array([
                [x_center - width / 2, y_center - height / 2],
                [x_center + width / 2, y_center - height / 2],
                [x_center + width / 2, y_center + height / 2],
                [x_center - width / 2, y_center + height / 2]
            ])

            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_corner_points = np.dot(corner_points - [x_center, y_center], rotation_matrix.T) + [x_center, y_center]

            distances = np.linalg.norm(rotated_corner_points - points, axis=1)
            return np.sum(distances ** 2)

        initial_guess = [0, 0, 1, 1, 0]
        result = minimize(objective_function, initial_guess, args=(point_list,), method='Nelder-Mead')
        x_center, y_center, width, height, angle = result.x
        corner_positions = np.array([
            [x_center - width / 2, y_center - height / 2],
            [x_center + width / 2, y_center - height / 2],
            [x_center + width / 2, y_center + height / 2],
            [x_center - width / 2, y_center + height / 2]
        ])

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_corner_positions = np.dot(corner_positions - [x_center, y_center], rotation_matrix.T) + [x_center, y_center]
        return cls.find_corner_dists(point_list, rotated_corner_positions)

    @classmethod
    def find_corner_dists(cls, true_corners, estimated_corners):
        def calculate_distance(point1, point2):
            return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

        min_distances = []

        for point1 in true_corners:
            min_distance = float('inf')
            for point2 in estimated_corners:
                distance = calculate_distance(point1, point2)
                min_distance = min(min_distance, distance)
            min_distances.append(min_distance)

        return min_distances


class SpatialPattern:
    """Spatial pattern, describing spatial relationship"""
    name = "spatial"
    _num_limit = [2, 2]  # spatial pattern are between 2 objects

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        sample_info = kwargs.get("sample_info", {"spatial_label": [0, 0, 0, 0]})

        enable_vis = False
        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses_pix = []
        anchor_sampled = True
        for id in obj_ids:
            if id != obj_id:
                if id in obj_poses_pix:
                    rel_obj_ids.append(id)
                    rel_obj_poses_pix.append(obj_poses_pix[id])
                else:
                    anchor_sampled = False

        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)
        pix_padding = int(0.1 * min(height, width))
        # compute anchor
        if len(rel_obj_poses_pix) > 0:
            assert len(rel_obj_poses_pix) == 1, "Only one anchor object is allowed!"
            anchor = [rel_obj_poses_pix[0][1], rel_obj_poses_pix[0][0]]
            raw_anchor = anchor.copy()  # for debug
        else:
            if anchor_sampled:
                # if no object is anchor, we use the center of the image
                anchor = [height/2, width/2]
                warnings.warn("No anchor provided; use center instead!")
            else:
                warnings.warn("Anchor object exists, but not sampled!")
                return prior, {}

        # parse spatial label
        spatial_label = list(sample_info["spatial_label"])  # [left, right, front, back]
        if spatial_label == [1, 0, 0, 0]:
            # left
            anchor[0] = np.max([anchor[0] - pix_padding, 0])
            prior[:, :int(anchor[0])] = 1.0
            spatial_str = "left"
            range_x = 0.15
            range_y = 0.01
            angle = 0
        elif spatial_label == [0, 1, 0, 0]:
            # right
            anchor[0] = np.min([anchor[0] + pix_padding, width - 1])
            prior[:, int(anchor[0]):] = 1.0
            spatial_str = "right"
            range_x = 0.15
            range_y = 0.01
            angle = 0
        elif spatial_label == [0, 0, 1, 0]:
            # front
            anchor[1] = np.min([anchor[1] + pix_padding, height - 1])
            prior[int(anchor[1]):, :] = 1.0
            spatial_str = "front"
            range_x = 0.01
            range_y = 0.15
            angle = np.pi / 2.0
        elif spatial_label == [0, 0, 0, 1]:
            # back
            anchor[1] = np.max([anchor[1] - pix_padding, 0])
            prior[:int(anchor[1]), :] = 1.0
            spatial_str = "back"
            range_x = 0.01
            range_y = 0.15
            angle = np.pi / 2.0
        elif spatial_label == [1, 0, 1, 0]:
            # left & front
            anchor[0] = np.max([anchor[0] - pix_padding, 0])
            anchor[1] = np.min([anchor[1] + pix_padding, height - 1])
            prior[int(anchor[1]):, :int(anchor[0])] = 1.0
            spatial_str = "left & front"
            range_x = 0.15
            range_y = 0.15
            angle = np.pi / 4.0
        elif spatial_label == [1, 0, 0, 1]:
            # left & back
            anchor[0] = np.max([anchor[0] - pix_padding, 0])
            anchor[1] = np.max([anchor[1] - pix_padding, 0])
            prior[:int(anchor[1]), :int(anchor[0])] = 1.0
            spatial_str = "left & back"
            range_x = 0.15
            range_y = 0.15
            angle = -np.pi / 4.0
        elif spatial_label == [0, 1, 1, 0]:
            # right & front
            anchor[0] = np.min([anchor[0] + pix_padding, width - 1])
            anchor[1] = np.min([anchor[1] + pix_padding, height - 1])
            prior[int(anchor[1]):, int(anchor[0]):] = 1.0
            spatial_str = "right & front"
            range_x = 0.15
            range_y = 0.15
            angle = np.pi / 4.0
        elif spatial_label == [0, 1, 0, 1]:
            # right & back
            anchor[0] = np.min([anchor[0] + pix_padding, width - 1])
            anchor[1] = np.max([anchor[1] - pix_padding, 0])
            prior[:int(anchor[1]), int(anchor[0]):] = 1.0
            spatial_str = "right & back"
            range_x = 0.15
            range_y = 0.15
            angle = -np.pi / 4.0
        else:
            raise NotImplementedError("Spatial label {} not implemented!".format(spatial_label))

        force_close = kwargs.get("force_close", True)
        if force_close:
            # small rectangle
            prior_close = np.zeros([height, width], dtype=np.float32)
            min_size = min(height, width)
            cv2.rectangle(prior_close, (int(anchor[0] - min_size * range_x), int(anchor[1] - min_size * range_y)),
                          (int(anchor[0] + min_size * range_x), int(anchor[1] + min_size * range_y)), 1.0, -1)
            prior = prior * prior_close

        if enable_vis:
            vis_prior = prior.copy()
            cv2.circle(vis_prior, (int(raw_anchor[0]), int(raw_anchor[1])), 1, 1.0, -1)
            cv2.imshow(f"prior-{spatial_str}", vis_prior)
            cv2.waitKey(1)

        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:spatial"
        pattern_info["obj_id"] = obj_id
        pattern_info["obj_ids"] = obj_ids
        pattern_info["spatial_label"] = spatial_label
        pattern_info["spatial_str"] = spatial_str
        pattern_info["angle"] = angle
        pattern_info["disable_collision_check"] = False
        # cv2.imshow("prior", prior)
        # cv2.waitKey(0)
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], pattern_info, **kwargs):
        """Check if obj poses meet the spatial pattern, spatial pattern is relevant with coordinate"""
        flip_xy = kwargs.get("flip_xy", False)
        if flip_xy:
            coordinate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        else:
            coordinate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        x_axis = coordinate[0]
        y_axis = coordinate[1]

        obj_id = pattern_info["obj_ids"][-1]  # the second one is to be checked
        obj_ids = pattern_info["obj_ids"]
        spatial_str = pattern_info["spatial_str"]
        # extract relative obj & poses
        rel_obj_ids = []
        rel_obj_poses = []
        for id in obj_ids:
            if id != obj_id:
                if id in obj_poses:
                    rel_obj_ids.append(id)
                    rel_obj_poses.append(obj_poses[id][:3])
                else:
                    return False
        rel_obj_pose = rel_obj_poses[0]
        obj_pose = obj_poses[obj_id][:3]
        pos_diff = obj_pose - rel_obj_pose
        # check spatial relationship
        spatial_label = pattern_info["spatial_label"]
        if spatial_label[0] == 1:
            # left
            if pos_diff.dot(x_axis) > 0:
                print(f"Spatial check failed: left; All: {spatial_str}; pose_diff: {pos_diff}.")
                return False
        elif spatial_label[1] == 1:
            # right
            if pos_diff.dot(x_axis) < 0:
                print(f"Spatial check failed: right; All: {spatial_str}; pose_diff: {pos_diff}.")
                return False
        elif spatial_label[2] == 1:
            # front
            if pos_diff.dot(y_axis) < 0:
                print(f"Spatial check failed: front; All: {spatial_str}; pose_diff: {pos_diff}.")
                return False
        elif spatial_label[3] == 1:
            # back
            if pos_diff.dot(y_axis) > 0:
                print(f"Spatial check failed: back; All: {spatial_str}; pose_diff: {pos_diff}.")
                return False
        return True

# Example: structformer has a pattern called dinner


class DataDrivenPattern:
    name = "data_driven"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        pattern_name = kwargs.get("pattern_name", "dinner")
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 1)
        obj_semantic_labels = kwargs.get("obj_semantic_labels", [])  # folk at left, plate at center, knife at right


# PATTERN DICT
PATTERN_DICT = {
    "uniform": Pattern,
    "rigid": RigidPattern,
    "line": LinePattern,
    "circle": CirclePattern,
    "rectangle": RectanglePattern,
    "tower": TowerPattern,
    "spatial": SpatialPattern
}
