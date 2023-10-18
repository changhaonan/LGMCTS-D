"""A pushing task that push an object to a region."""

from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
import numpy as np
import warnings
from copy import deepcopy
import random
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
            debug: bool = False,):
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
        self.base_color_list = [TexturePedia.lookup_color_by_name(color) for color in base_color_list]
        self.obs_img_size = obs_img_size
        self.ee = Spatula  # override the ee with spatula
        self.primitive = Push()  # override the primitive with push
        # task specific
        self.objs_to_push = []
        self.in_region = None  # an region object


    def reset(self, env):
        """Reset the task"""
        super().reset(env)
        # add base object
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

        obj_id, base_urdf_full_path = pybullet_utils.add_any_object(
            env=env,
            obj_entry=sampled_base_obj,
            pose=base_pose,
            size=base_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )

        pybullet_utils.p_change_texture(obj_id, sampled_base_obj_texture, env.client_id)
        pybullet_utils.add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=sampled_base_obj,
            texture_entry=sampled_base_obj_texture,
        )
        # add objects
        num_objs_to_add = 1
        selected_objs = self.rng.choice(self.obj_list, size=num_objs_to_add, replace=True)
        selected_colors = self.rng.choice(self.color_list, size=num_objs_to_add, replace=True)
        self.add_objects_to_pattern(
            env=env,
            objs=selected_objs,
            colors=selected_colors,
            pattern_prior=None,
        )

    def check_success(self):
        """Check if the task is finished"""
        return ResultTuple(success=False, failure=False, distance=None)

    def oracle_action(self, env):
        """Get oracle action Given current states"""
        return None