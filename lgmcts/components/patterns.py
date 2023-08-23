"""Define patterns here"""
from __future__ import annotations
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import math
import warnings
from scipy.optimize import curve_fit
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
            x1 = rel_obj_poses_pix[0][1]
            y1 = rel_obj_poses_pix[0][0]
        else:
            # if more than one object is sampled, we generate a line based on the objects
            x0 = rel_obj_poses_pix[0][1]
            y0 = rel_obj_poses_pix[0][0]
            x1 = rel_obj_poses_pix[1][1]
            y1 = rel_obj_poses_pix[1][0]
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


class CirclePattern(Pattern):
    """Circle pattern, obj poses should formulate a circle"""
    name = "circle"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate circle prior"""
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

        height, width = img_size
        prior = np.zeros(img_size, dtype=np.float32)

        if len(rel_obj_ids) == 0:
            center_x = rng.integers(0, width)
            center_y = rng.integers(0, height)
            radius = rng.integers(2, min(center_x, center_y, height - center_x - 1, width - center_y - 1))  # Random radius within limits
        elif len(rel_obj_ids) == 1:
            x0, y0 = rel_obj_poses_pix[0]
            radius = int(np.sqrt(x0**2 + y0**2))  # Distance origin to point is the radius
            center_x, center_y = x0, y0
        else:
            # if more than one object is sampled, we generate a line based on the objects
            points = np.array(rel_obj_poses_pix)
            # Find the minimum enclosing circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(points)
            center_x = int(center_x)
            center_y = int(center_y)
            radius = int(radius)

        # Draw the circle on the image
        cv2.circle(prior, (center_x, center_y), radius, 1.0, thickness)
        
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:circle"
        pattern_info["center_pixel"] = [int(center_x), int(center_y)]
        pattern_info["radius"] = radius
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if obj poses meet a circle pattern"""
        assert "pattern_info" in kwargs, "Pattern info must be provided!"
        pattern_info = kwargs["pattern_info"]
        
        # Check if p2c distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        center = np.array(pattern_info["center_pixel"])
        radius = pattern_info["radius"]

        # assemble obj_poses
        obj_poses_pattern = []
        for obj_id in pattern_info["obj_ids"]:
            obj_poses_pattern.append(obj_poses[obj_id][:3])
        obj_poses_pattern = np.vstack(obj_poses_pattern)
        
        # Calculate distances from object poses to the circle's center
        dists = np.linalg.norm(cls.dist_rad(center, radius, obj_poses_pattern[:, :2]))
        
        return not (np.max(dists) > threshold)
    
    @classmethod
    def dist_rad(cls, center, radius, point_list):
        distances = []
        for point in point_list:
            distance = abs(cls.euclidean_distance(center, point) - radius)
            distances.append(distance)
        return distances
    
    @classmethod
    def euclidean_distance(cls, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    

class RectanglePattern(Pattern):
    """Rectangle pattern, obj poses should formulate a rectangle"""
    name = "rectangle"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate rectangle prior"""
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

        height, width = img_size
        prior = np.zeros(img_size, dtype=np.float32)
        
        if len(rel_obj_ids) == 0:
            x0 = rng.integers(0, width - 10)
            y0 = rng.integers(0, height - 10)
            x1 = rng.integers(x0 + 10, width)
            y1 = rng.integers(y0 + 10, height)
        elif len(rel_obj_ids) == 1:
            #use rel_obj_poses_pix[0] as bottom left corner
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rng.integers(x0 + 10, width)
            y1 = rng.integers(y0 + 10, height)
        elif len(rel_obj_ids) == 2:
            #use bottom left and top right to create rectangle
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rel_obj_poses_pix[1][0]
            y1 = rel_obj_poses_pix[1][1]
        else:
            #fit rectangle all the given points using boundingRect (possible to use rotatedRect for rotation)
            points_array = np.array(rel_obj_poses_pix)
            x0, y0, width, height = cv2.boundingRect(points_array)
            x1 = x0 + width
            y1 = y0 + height

        # Draw the rectangle on the image
        cv2.rectangle(prior, (x0, y0), (x1, y1), 1.0, thickness)
        
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:rectangle"
        pattern_info["top_left_pixel"] = [int(x0), int(y0)]
        pattern_info["bottom_right_pixel"] = [int(x1), int(y1)]
        
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
            obj_poses_pattern.append(obj_poses[obj_id][:3])
        obj_poses_pattern = np.vstack(obj_poses_pattern)

        # Calculate rectangle boundary coordinates
        top_left = np.array(pattern_info["top_left_pixel"])
        bottom_right = np.array(pattern_info["bottom_right_pixel"])
        top_right, bottom_left = cls.calculate_other_corners(top_left, bottom_right)
        
        # Calculate distances from object poses to the circle's center
        dists = []
        for point in obj_poses_pattern[:, :2]:
            min_distances = cls.distances_to_borders(point, [top_left, top_right, bottom_right, bottom_left])
            min_distance = min(min_distances)
            dists.append(min_distance)
        
        return not (np.max(np.linalg.norm(dists)) > threshold)

    @classmethod
    def line_equation(cls, p1, p2):
        """Calculate equations of lines"""
        # Calculate slope (m) and y-intercept (b)
        if p2[0] - p1[0] == 0:
            slope = float('inf')  # Vertical line
            y_intercept = p1[0]
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            y_intercept = p1[1] - slope * p1[0]
        return slope, y_intercept

    @classmethod
    def point_to_line_distance(cls, point, line):
        """Calculate distance between point and line"""
        distance = abs(line[0] * point[0] - point[1] + line[1]) / math.sqrt(line[0] ** 2 + 1)
        return distance

    @classmethod
    def distances_to_borders(cls, point, rectangle_corners):
        """Calculate distances from points to rectangle borders"""
        distances = []
        for i in range(4):
            line = cls.line_equation(rectangle_corners[i], rectangle_corners[(i+1) % 4])
            distance = cls.point_to_line_distance(point, line)
            distances.append(distance)
        return distances

    @classmethod
    def calculate_other_corners(cls, top_left, bottom_right):
        x1, y1 = top_left
        x2, y2 = bottom_right
        top_right = [x2, y1]
        bottom_left = [x1, y2]
        return top_right, bottom_left


class SineCurvePattern(Pattern):
    """Sine curve pattern, obj poses should formulate a sine curve"""
    name = "sine_curve"

    @classmethod
    def gen_prior(cls, img_size, rng, frequency=0.1, amplitude=0.5, **kwargs):
        """Generate sine curve prior"""
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
        
        height, width = img_size
        prior = np.zeros(img_size, dtype=np.float32)
        
        if len(rel_obj_ids) == 0:
            x_vals = np.linspace(0, width - 1, width)
            y_vals = (np.sin(x_vals * frequency) + 1) * (amplitude * (height - 1))
        else:
            #fit sine curve to points here using scipy curvefit
            points = np.array(rel_obj_poses_pix)

            def sine_curve(x, A, f, phi, c):
                return A * np.sin(2 * np.pi * f * x + phi) + c
            
            initial_guess = [amplitude, frequency, 0, 0]
            popt, _ = curve_fit(sine_curve, points[:, 0], points[:, 1], p0=initial_guess)
            
            x_vals = np.linspace(0, width - 1, width)
            y_vals = sine_curve(x_vals, *popt)
        
        # Draw the sine curve on the image
        for i in range(width):
            x, y = int(x_vals[i]), int(y_vals[i])
            prior[y, x] = 1.0

        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:sine_curve"
        pattern_info["control_points"] = list(zip(x_vals.astype(int), y_vals.astype(int)))
        pattern_info["frequency"] = frequency
        pattern_info["amplitude"] = amplitude 
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], pattern_info, **kwargs):
        """Check if obj poses meet the sine curve pattern"""
        control_points = pattern_info["control_points"]
        
        #assign object poses
        obj_poses_pattern = []
        for obj_id in pattern_info["obj_ids"]:
            obj_poses_pattern.append(obj_poses[obj_id][:3])

        # Calculate distances between object poses and control points on the sine curve
        dists = []
        for obj_pose in obj_poses_pattern:
            min_dist = float("inf")
            for cp_x, cp_y in control_points:
                dist = np.linalg.norm(obj_pose[:2] - np.array([cp_x, cp_y]))
                min_dist = min(min_dist, dist)
            dists.append(min_dist)
        
        # Check if the minimum distance from each object to the sine curve is below the threshold
        threshold = pattern_info.get("threshold", 0.1)
        return not (np.max(np.linalg.norm(dists)) > threshold)


#TODO: we need to add more patterns here, e.g. circle, rectangle, spatial, etc. @Alex


## PATTERN DICT

PATTERN_DICT = {
    "line": LinePattern,
    "circle": CirclePattern,
    "rectangle": RectanglePattern,
    "sine": SineCurvePattern
}