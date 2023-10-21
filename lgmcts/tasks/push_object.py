"""A pushing task that push an object to a region.
FIXME: Currently, have a lot of hacks, just for testing.
"""
from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
import numpy as np
import warnings
from copy import deepcopy
import random
from shapely.geometry import Point, Polygon
import lgmcts.utils.misc_utils as utils
import lgmcts.utils.spatial_utils as spatial_utils
import lgmcts.utils.pybullet_utils as pybullet_utils
from lgmcts.tasks import BaseTask
from lgmcts.components.encyclopedia import ObjPedia, TexturePedia
from lgmcts.components.placeholders import PlaceholderText
from lgmcts.components.prompt import PromptGenerator
from lgmcts.components.obj_selector import ObjectSelector
from lgmcts.components.attribute import COMPARE_DICT, EqualRel, DifferentRel, SmallerRel, BiggerRel
from lgmcts.components.patterns import PATTERN_DICT
from lgmcts.components.end_effectors import Spatula
from lgmcts.components.action_primitives import Push
from lgmcts.env import seed


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None


class Region:
    """Region object; determining if object has reached the region"""

    def __init__(self, bound_points) -> None:
        self.bound_points = bound_points
        self.region = Polygon(bound_points)
        self.center = np.mean(bound_points, axis=0)

    def __contains__(self, pose: np.ndarray) -> bool:
        """Check if the pose is in the region"""
        pose_xy = pose[:2]
        point = Point(pose_xy)
        return self.region.contains(point)


class PushObject(BaseTask):
    """Push task to push an object to a region."""

    task_name = f"push_object_{seed}"

    def __init__(
        self,
        # ==== task specific ====
        max_num_obj: int = 10,
        max_num_pattern: int = 2,
        stack_prob: float = 0.0,
        pattern_types: list[str] = ["line", "circle"],
        obj_list: list[str] | None = None,
        color_list: list[str] | None = None,
        base_obj_list: list[str] | None = None,
        base_color_list: list[str] | None = None,
        # ==== general ====
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        super().__init__(
            modalities=["rgb"],
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            seed=seed,
            debug=debug,
        )
        # general
        self.max_num_obj = max_num_obj
        self.obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in obj_list]
        self.color_list = [TexturePedia.lookup_color_by_name(color) for color in color_list]
        self.base_obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in base_obj_list]
        self.base_color_list = [
            TexturePedia.lookup_color_by_name(color) for color in base_color_list
        ]
        self.obs_img_size = obs_img_size
        self.ee = Spatula  # override the ee with spatula
        self.primitive = Push(
            rest_height=0.005, operation_height=0.005
        )  # override the primitive with push
        # task specific
        self.objs_to_push = []
        self.regions_to_push = []

    def reset(self, env):
        """Reset the task"""
        super().reset(env)
        self.objs_to_push = []
        self.regions_to_push = []
        # add base object; push target
        sampled_base_obj = self.rng.choice(self.base_obj_list).value
        sampled_base_obj_texture = self.rng.choice(self.base_color_list).value

        base_obj_sampled_size = self.rng.uniform(
            low=sampled_base_obj.size_range.low,
            high=sampled_base_obj.size_range.high,
        )
        base_obj_sampled_size *= 1.35

        base_pos_x = env.bounds[0, 0] + 0.16
        base_pos_y = 0
        base_rot = utils.eulerXYZ_to_quatXYZW((0, 0, -np.pi / 2))
        base_pos = (base_pos_x, base_pos_y, base_obj_sampled_size[2] / 2)
        base_pose = (base_pos, base_rot)

        base_obj_id, base_urdf_full_path = pybullet_utils.add_any_object(
            env=env,
            obj_entry=sampled_base_obj,
            pose=base_pose,
            size=base_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )

        pybullet_utils.p_change_texture(base_obj_id, sampled_base_obj_texture, env.client_id)
        pybullet_utils.add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=base_obj_id,
            object_entry=sampled_base_obj,
            texture_entry=sampled_base_obj_texture,
        )
        base_aabb = pybullet_utils.get_obj_aabb(env, base_obj_id)
        min_xyz, max_xyz = base_aabb
        bound_points = np.array(
            [
                [min_xyz[0], min_xyz[1]],
                [min_xyz[0], max_xyz[1]],
                [max_xyz[0], max_xyz[1]],
                [max_xyz[0], min_xyz[1]],
            ]
        )
        self.regions_to_push.append(Region(bound_points))

        # add objects to push
        # FIXME: because of push bug, I need to generate the object to x > 0, y > 0 region
        prior_size = env.ws_map_size
        prior = np.zeros(prior_size, dtype=np.float32)
        prior[env.ws_map_size[0] // 2 :, :] = 1.0
        # add 2 objects
        num_objs_to_add = 2
        selected_objs = self.rng.choice(self.obj_list, size=num_objs_to_add, replace=True)
        selected_colors = self.rng.choice(self.color_list, size=num_objs_to_add, replace=True)
        self.objs_to_push, _ = self.add_objects_to_pattern(
            env=env,
            objs=selected_objs,
            colors=selected_colors,
            pattern_prior=prior,
        )

    def check_success(self, env):
        """Check if the task is finished"""
        # check if all objects are in the region
        all_obj_in_region = True
        for obj in self.objs_to_push:
            if not self.check_obj_in_region(env, obj):
                all_obj_in_region = False
                break
        success = all_obj_in_region
        return ResultTuple(success=success, failure=not success, distance=None)

    def check_obj_in_region(self, env, obj):
        obj_pose = pybullet_utils.get_obj_pose(env, obj)
        obj_pose = np.array(obj_pose[0])
        if np.abs(obj_pose[1]) < 0.1:  # check y
            return True

    def oracle_action(self, env):
        """Get oracle action Given current states"""
        # FIXME: currently has a bug, can only push object in (x > 0, y > 0) region
        prep_height = 0.3
        action_height = 0.005
        action_list = []
        for obj in self.objs_to_push:
            if self.check_obj_in_region(env, obj):
                # not push obj that is already in the region
                continue
            obj_pose = pybullet_utils.get_obj_pose(env, obj)
            obj_position = np.array(obj_pose[0])
            # randomly select an region to push
            region_idx = self.rng.choice(len(self.regions_to_push))
            region = self.regions_to_push[region_idx]
            # compute step size
            min_movement = 0.03
            num_movement = int(
                np.ceil(np.linalg.norm(region.center - obj_position[:2]) / min_movement)
            )
            step_size = 1.0 / float(num_movement)
            prev_position = np.zeros(2, dtype=np.float32)
            # push
            scale_list = list(np.arange(-0.2, 1.0, step_size))
            for i, s in enumerate(scale_list):
                position = region.center * s + obj_position[:2] * (1.0 - s)
                # assmeble
                position_3d = np.array([position[0], position[1], action_height])
                prev_position_3d = np.array([prev_position[0], prev_position[1], action_height])
                if i == 1:
                    prev_position_3d = np.array([obj_position[0], obj_position[1], prep_height])
                action = {
                    "pose0_position": prev_position_3d,
                    "pose0_rotation": np.array([0.0, 0.0, 0.0, 1.0]),
                    "pose1_position": position_3d,
                    "pose1_rotation": np.array([0.0, 0.0, 0.0, 1.0]),
                }
                if i > 0:
                    action_list.append(action)
                prev_position = position.copy()
            break  # only push one object one time
        return action_list
