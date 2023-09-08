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
    "circle": {
        "radius": {
            "L" : [0.5, 0.7],
            "M" : [0.3, 0.5],
            "S" : [0.1, 0.3]
        }
    }
}


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
        height, width = img_size[0], img_size[1]
        prior = np.zeros([height, width], dtype=np.float32)
        if len(rel_obj_ids) == 0:
            # first point can be anywhere
            return np.ones_like(prior), {"type": "pattern:line", "position_pixel": [0, 0, width-1, height-1], "rotation": [0, 0, 0]}
        elif len(rel_obj_ids) == 1:
            # random pixel
            x0 = 0 if rel_obj_poses_pix[0][1] == 0 else width-1
            y0 = rel_obj_poses_pix[0][0]  #HACK: line will be parallel to x-axis
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
            start_point = (int(x0), 0)
            end_point = (int(x0), height-1)
            cv2.line(prior, start_point, end_point, 1.0, thickness)
        elif y1 - y0 == 0:  # horizontal line
            start_point = (0, int(y0))
            end_point = (width-1, int(y0))
            cv2.line(prior, start_point, end_point, 1.0, thickness)
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
        # cv2.waitKey(0.01)
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
        status = not(np.max(dists) > threshold)
        if not status:
            print("Line pattern check failed!")
        return status

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
        rel_size = kwargs.get("rel_size", "M")

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
            #radius = rng.integers(0, min([center_x, center_y, width - center_x - 1, height - center_y - 1]))
            max_rad = min([height, width])
            scaled_min_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][0] * max_rad
            scaled_max_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][1] * max_rad
            radius = rng.integers(scaled_min_rad, scaled_max_rad)
            
        elif len(rel_obj_ids) == 1:
            x0, y0 = rel_obj_poses_pix[0]
            max_rad = int(min([height, width])/2)
            scaled_min_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][0] * max_rad
            scaled_max_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][1] * max_rad
            radius = rng.integers(scaled_min_rad, scaled_max_rad)
            angle_direction = rng.integers(0,8)
            angle_degrees = angle_direction * 45
            angle_radians = math.radians(angle_degrees)
            new_x = int(x0 + radius * math.cos(angle_radians))
            new_y = int(y0 - radius * math.sin(angle_radians))
            center_x, center_y = new_x, new_y
        elif len(rel_obj_ids) == 2:
            x0, y0 = rel_obj_poses_pix[0]
            x1, y1 = rel_obj_poses_pix[1]
            
            midpoint = [(x0 + x1) / 2, (y0 + y1) / 2]
            distances = [
                midpoint[0],             
                width - midpoint[0],      
                midpoint[1],               
                height - midpoint[1]      
            ]
            max_distance = max(distances)
            chord_len = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)/2

            max_rad = int(-((-4*(max_distance**2)) - (chord_len**2))/(8*max_distance))
            scaled_min_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][0] * max_rad
            scaled_max_rad = PATTERN_CONSTANTS["circle"]["radius"][rel_size][1] * max_rad
            radius = rng.integers(scaled_min_rad, scaled_max_rad)

            direction = rng.choice([-1, 1])
            distance = np.sqrt(radius**2 - ((x1 - x0)**2 + (y1 - y0)**2) / 4)

            center_vector = [direction * (y0 - y1) * distance / (2 * np.sqrt((x1 - x0)**2 + (y1 - y0)**2)),
                            direction * (x1 - x0) * distance / (2 * np.sqrt((x1 - x0)**2 + (y1 - y0)**2))]

            center_x = midpoint[0] + center_vector[0]
            center_y = midpoint[1] + center_vector[1]

        else:
            # if more than one object is sampled, we generate a circle based on the objects
            points = np.array(rel_obj_poses_pix)
            # Find the minimum enclosing circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(points)
            center_x = int(center_x)
            center_y = int(center_y)
            radius = int(radius)

        # Draw the circle on the image
        cv2.circle(prior, (center_x, center_y), radius, 1.0, thickness)
        #cv2.imshow("prior", prior)
        #cv2.waitKey(500)
        
        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:circle"
        pattern_info["center_pixel"] = [center_x, center_y]
        pattern_info["radius"] = radius
        pattern_info["obj_ids"] = obj_ids
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], **kwargs):
        """Check if obj poses meet a circle pattern"""
        assert "pattern_info" in kwargs, "Pattern info must be provided!"
        pattern_info = kwargs["pattern_info"]
        
        # Check if p2c distance exceeds threshold
        threshold = pattern_info.get("threshold", 0.1)
        # assemble obj_poses
        obj_poses_pattern = []
        for obj_id in pattern_info["obj_ids"]:
            obj_poses_pattern.append(obj_poses[obj_id][:3])
        
        if len(obj_poses_pattern) < 4:
            return True
        else:
            # Calculate distances from object poses to the circle's center
            dists = cls.dist_rad(np.array([[sublist[0], sublist[1]] for sublist in obj_poses_pattern]))
            #print(dists)
            
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
            x0 = rng.integers(0, width)
            y0 = rng.integers(0, height)
            x1 = rng.integers(0, width)
            y1 = rng.integers(0, height)
            max_len = rng.integers(0, min([width, height]))
            length = rng.integers(2, max_len)
            same_edge = rng.integers(0, 2) # 0 = both corners along edge, 1 = corners opposite
            same_edge = 0
            if same_edge == 0:
                x2, y2, x3, y3 = cls.calc_same_edge_corners(x0, y0, x1, y1, rng, length)
            else:
                x2 = x0
                y2 = y1
                x3 = x1
                y3 = y0
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            p3 = [x3, y3]
            diagonal, point0, point1 = cls.find_diagonal(p0, p1, p2)
            rect_points = [point0, diagonal, point1, p3]
        elif len(rel_obj_ids) == 1:
            #use rel_obj_poses_pix[0] as a random corner
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rng.integers(0, width)
            y1 = rng.integers(0, height)
            max_len = rng.integers(0, min([width, height]))
            length = rng.integers(2, max_len)
            same_edge = rng.integers(0, 2) # 0 = both corners along edge, 1 = corners opposite
            same_edge = 0
            if same_edge == 0:
                x2, y2, x3, y3 = cls.calc_same_edge_corners(x0, y0, x1, y1, rng, length)
            else:
                x2 = x0
                y2 = y1
                x3 = x1
                y3 = y0
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            p3 = [x3, y3]
            diagonal, point0, point1 = cls.find_diagonal(p0, p1, p2)
            rect_points = [point0, diagonal, point1, p3]
        elif len(rel_obj_ids) == 2:
            #use bottom left and top right to create rectangle
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rel_obj_poses_pix[1][0]
            y1 = rel_obj_poses_pix[1][1]
            max_len = rng.integers(0, min([width, height]))
            length = rng.integers(2, max_len)
            same_edge = rng.integers(0, 2) # 0 = both corners along edge, 1 = corners opposite
            same_edge = 0
            if same_edge == 0:
                x2, y2, x3, y3 = cls.calc_same_edge_corners(x0, y0, x1, y1, rng, length)
            else:
                x2 = x0
                y2 = y1
                x3 = x1
                y3 = y0
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            p3 = [x3, y3]
            diagonal, point0, point1 = cls.find_diagonal(p0, p1, p2)
            rect_points = [point0, diagonal, point1, p3]

        elif len(rel_obj_ids) == 3:
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rel_obj_poses_pix[1][0]
            y1 = rel_obj_poses_pix[1][1]
            x2 = rel_obj_poses_pix[2][0]
            y2 = rel_obj_poses_pix[2][1]
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            diagonal, point0, point1 = cls.find_diagonal(p0, p1, p2)
            x3, y3 = cls.find_fourth_point(point0, diagonal, point1)
            point4 = [x3, y3]
            rect_points = [point0, diagonal, point1, point4]
        elif len(rel_obj_ids) == 4:
            x0 = rel_obj_poses_pix[0][0]
            y0 = rel_obj_poses_pix[0][1]
            x1 = rel_obj_poses_pix[1][0]
            y1 = rel_obj_poses_pix[1][1]
            x2 = rel_obj_poses_pix[2][0]
            y2 = rel_obj_poses_pix[2][1]
            x3 = rel_obj_poses_pix[3][0]
            y3 = rel_obj_poses_pix[3][1]
            p0 = [x0, y0]
            p1 = [x1, y1]
            p2 = [x2, y2]
            p3 = [x3, y3]
            diagonal, point0, point1 = cls.find_diagonal(p0, p1, p2)
            rect_points = [point0, diagonal, point1, p3]

        # Draw the rectangle on the image
        #use lines for visualization but circles to restrain poses to corners
        #cv2.line(prior, rect_points[0], rect_points[1], (255,255,255), thickness)
        #cv2.line(prior, rect_points[1], rect_points[2], (255,255,255), thickness)
        #cv2.line(prior, rect_points[2], rect_points[3], (255,255,255), thickness)
        #cv2.line(prior, rect_points[3], rect_points[0], (255,255,255), thickness)
        cv2.circle(prior, rect_points[0], radius=thickness, color=(255,255,255), thickness=-1)
        cv2.circle(prior, rect_points[1], radius=thickness, color=(255,255,255), thickness=-1)
        cv2.circle(prior, rect_points[2], radius=thickness, color=(255,255,255), thickness=-1)
        cv2.circle(prior, rect_points[3], radius=thickness, color=(255,255,255), thickness=-1)


        #cv2.imshow("prior", prior)
        #cv2.waitKey(5000)

        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:rectangle"
        pattern_info["corners"] = rect_points
        pattern_info["obj_ids"] = obj_ids
        
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
        if len(obj_poses_pattern) < 4:
            warnings.warn("not enough points")
        else:
            dists = cls.dist_corners([[sublist[0], sublist[1]] for sublist in obj_poses_pattern])
            return not(np.max(dists) > threshold)

    @classmethod
    def calc_same_edge_corners(cls, x0, y0, x1, y1, rng, length):
        dx = x1 - x0
        dy = y1 - y0

        px = -dy
        py = dx

        magnitude = math.sqrt(px**2 + py**2)
        px /= magnitude
        py /= magnitude

        direction = rng.choice([-1, 1])


        x2 = int(x0 + direction * length * px)
        y2 = int(y0 + direction * length * py)
        x3 = int(x1 + direction * length * px)
        y3 = int(y1 + direction * length * py)
        return x2, y2, x3, y3
    
    @classmethod
    def find_fourth_point(cls, p1, p2, p3):
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        side1 = p3 - p2
        p4 = p1 + side1
        return p4[0], p4[1]


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

            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
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

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
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


class SineCurvePattern(Pattern):
    """Sine curve pattern, obj poses should formulate a sine curve"""
    name = "sine_curve"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        """Generate sine curve prior"""
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        frequency = kwargs.get("frequency", 0.1)
        amplitude = kwargs.get("amplitude", 0.5)
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
            y_vals = np.linspace(0, width - 1, width)
            x_vals = (np.sin(x_vals * frequency) + 1) * (amplitude * (height - 1))
        else:
            #fit sine curve to points here using scipy curvefit
            points = np.array(rel_obj_poses_pix)

            def sine_curve(y, phi, c, A = amplitude, f = frequency):
                return A * np.sin(2 * np.pi * f * y + phi) + c
            
            initial_guess = [0, 0]
            popt, _ = curve_fit(sine_curve, np.array(points[:, 1]), np.array(points[:, 0]), p0=initial_guess)
            x_vals = np.linspace(0, height - 1, height)
            y_vals = sine_curve(x_vals, *popt)
        
        # Draw the sine curve on the image
        for i in range(height):
            x, y = int(x_vals[i]), int(y_vals[i])
            if y > width - 1 or y < 0:
                continue
            prior[x, y] = 1.0

        # cv2.imshow("prior", prior)
        # cv2.waitKey(5000)

        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:sine_curve"
        pattern_info["control_points"] = list(zip(x_vals/width, y_vals/width))
        pattern_info["frequency"] = frequency
        pattern_info["amplitude"] = amplitude 
        pattern_info["obj_ids"] = obj_ids
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


class SpatialPattern:
    """Spatial pattern, describing spatial relationship"""
    name = "spatial"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        sample_info = kwargs.get("sample_info", {"spatial_label": [0, 0, 0, 0]})
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
        # compute anchor
        if len(rel_obj_poses_pix) > 0:
            assert len(rel_obj_poses_pix) == 1, "Only one anchor object is allowed!"
            anchor = [rel_obj_poses_pix[0][1], rel_obj_poses_pix[0][0]]
        else:
            if anchor_sampled:
                # if no object is anchor, we use the center of the image
                anchor = [height/2, width/2]
            else:
                warnings.warn("Anchor object exists, but not sampled!")
                return prior, {}

        # parse spatial label 
        spatial_label = list(sample_info["spatial_label"])  # [left, right, front, back]
        if spatial_label == [1, 0, 0, 0]:
            # left
            anchor[0] = np.max([anchor[0] - 1, 0])
            prior[:, :int(anchor[0])] = 1.0
            spatial_str = "left"
        elif spatial_label == [0, 1, 0, 0]:
            # right
            anchor[0] = np.min([anchor[0] + 1, width - 1])
            prior[:, int(anchor[0]):] = 1.0
            spatial_str = "right"
        elif spatial_label == [0, 0, 1, 0]:
            # front
            anchor[1] = np.min([anchor[1] + 1, height - 1])
            prior[int(anchor[1]):, :] = 1.0
            spatial_str = "front"
        elif spatial_label == [0, 0, 0, 1]:
            # back
            anchor[1] = np.max([anchor[1] - 1, 0])
            prior[:int(anchor[1]), :] = 1.0
            spatial_str = "back"
        elif spatial_label == [1, 0, 1, 0]:
            # left & front
            anchor[0] = np.max([anchor[0] - 1, 0])
            anchor[1] = np.min([anchor[1] + 1, height - 1])
            prior[int(anchor[1]):, :int(anchor[0])] = 1.0
            spatial_str = "left & front"
        elif spatial_label == [1, 0, 0, 1]:
            # left & back
            anchor[0] = np.max([anchor[0] - 1, 0])
            anchor[1] = np.max([anchor[1] - 1, 0])
            prior[:int(anchor[1]), :int(anchor[0])] = 1.0
            spatial_str = "left & back"
        elif spatial_label == [0, 1, 1, 0]:
            # right & front
            anchor[0] = np.min([anchor[0] + 1, width - 1])
            anchor[1] = np.min([anchor[1] + 1, height - 1])
            prior[int(anchor[1]):, int(anchor[0]):] = 1.0
            spatial_str = "right & front"
        elif spatial_label == [0, 1, 0, 1]:
            # right & back
            anchor[0] = np.min([anchor[0] + 1, width - 1])
            anchor[1] = np.max([anchor[1] - 1, 0])
            prior[:int(anchor[1]), int(anchor[0]):] = 1.0
            spatial_str = "right & back"
        else:
            raise NotImplementedError("Spatial label {} not implemented!".format(spatial_label))

        # Pattern info
        pattern_info = {}
        pattern_info["type"] = "pattern:spatial"
        pattern_info["obj_id"] = obj_id
        pattern_info["obj_ids"] = obj_ids
        pattern_info["spatial_label"] = spatial_label
        pattern_info["spatial_str"] = spatial_str
        # cv2.imshow("prior", prior)
        # cv2.waitKey(0)
        return prior, pattern_info

    @classmethod
    def check(cls, obj_poses: dict[int, np.ndarray], pattern_info, **kwargs):
        """Check if obj poses meet the spatial pattern, spatial pattern is relevant with coordinate"""
        coordinate = kwargs.get("coordinate", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
        x_axis = coordinate[0]
        y_axis = coordinate[1]

        obj_id = pattern_info["obj_ids"][-1]  # the second one is to be checked
        obj_ids = pattern_info["obj_ids"]
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
                print("Spatial check failed: left")
                return False
        elif spatial_label[1] == 1:
            # right
            if pos_diff.dot(x_axis) < 0:
                print("Spatial check failed: right")
                return False
        elif spatial_label[2] == 1:
            # front
            if pos_diff.dot(y_axis) < 0:
                print("Spatial check failed: front")
                return False
        elif spatial_label[3] == 1:
            # back
            if pos_diff.dot(y_axis) > 0:
                print("Spatial check failed: back")
                return False
        return True

##TODO: Arbitrary pattern @Alex

#Example: structformer has a pattern called dinner

class DataDrivenPattern:
    name = "data_driven"

    @classmethod
    def gen_prior(cls, img_size, rng, **kwargs):
        pattern_name = kwargs.get("pattern_name", "dinner")
        obj_poses_pix = kwargs.get("obj_poses_pix", {})
        obj_id = kwargs.get("obj_id", -1)
        obj_ids = kwargs.get("obj_ids", [])
        thickness = kwargs.get("thickness", 1)
        obj_semantic_labels = kwargs.get("obj_semantic_labels", []) #folk at left, plate at center, knife at right




## PATTERN DICT

PATTERN_DICT = {
    "line": LinePattern,
    "circle": CirclePattern,
    "rectangle": RectanglePattern,
    "sine": SineCurvePattern,
    "spatial": SpatialPattern
}

## Test code
#if __name__ == "__main__":
#   # test spatial
#    spatial_label = [0, 1, 0, 1]
#    anchor = [50, 50]
#    img_size = [200, 200]
#    prior, pattern_info = SpatialPattern.gen_prior(img_size, None, spatial_label=spatial_label, anchor=anchor)
#    cv2.imshow("prior", prior)
#    cv2.waitKey(0)