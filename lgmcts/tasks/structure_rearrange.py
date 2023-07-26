from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
import lgmcts.utils.misc_utils as utils
from lgmcts.tasks import BaseTask
from lgmcts.encyclopedia import ObjPedia, TexturePedia
from lgmcts.placeholders import PlaceholderText, PlaceholderObj, PlaceholderScene


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None

    
class StructureRearrange(BaseTask):
    task_name = "structure_rearrange"

    def __init__(
        self, 
        # ==== task specific ====
        max_num_obj: int = 6,
        stack_prob: float = 0.0,
        pattern_types: list[str] = ["line", "circle"],
        obj_express_types: Literal["name", "image"] = "name",
        obj_list: list[str] | None = None,
        color_list: list[str] | None = None,
        # ==== general ====
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,):
        
        task_meta = {
            "max_num_obj": max_num_obj,
        }
        placeholder_expression = {
            f"obj_{i}" : {
                "type": obj_express_types,
            }
            for i in range(1, max_num_obj + 1)
        }

        # template
        prompt_template = [
            "Set {objs} to {pattern}",
        ]
        super().__init__(
            prompt_template=prompt_template,
            placeholder_expression=placeholder_expression,
            modalities=["rgb"],
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            seed=seed,
            debug=debug,
        )
        self.max_num_obj = max_num_obj
        self.stack_prob = stack_prob
        self.pattern_types = pattern_types
        self.obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in obj_list]
        self.color_list = [TexturePedia.lookup_color_by_name(color) for color in color_list]
        # 
        self.obs_img_size = obs_img_size
        # template

    def update_goals(self):
        # Three stages
        self.progress += 1
        if self.progress > 2:
            self.progress = 0

    def update_env(self, env):
        if self.progress == 0:
            # randomize the pattern type
            pattern_type = env.rng.choice(self.pattern_types)
            self.set_objects_to_pattern(env, pattern_type, False, self.stack_prob)  # Structured Goal
        elif self.progress == 1:
            self.set_objects_to_random(env, True, self.stack_prob) # Random Init
        
        env.wait_until_settle()

    def check_success(self):
        if self.progress == 0 or self.progress == 1:
            return ResultTuple(success=False, failure=True, distance=None)
        elif self.progress == 2:
            return ResultTuple(success=True, failure=False, distance=None)
        else:
            raise ValueError("Invalid progress value")

    def set_objects_to_pattern(self, env, pattern_type: str, use_existing=False, stack_prob=0.0):
        """Set objects to a line, use_existing decides whether to add new object or not"""
        line_pattern = utils.gen_random_pattern(pattern_type, env.occupy_size, env.rng)
        # Add object
        if not use_existing:
            env.reset()  # Clear all objects
            for i in range(self.max_num_obj):
                env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=line_pattern,
                    stack_prob=stack_prob,
                )
        else:
            raise NotImplementedError("Not implemented yet")

    def set_objects_to_random(self, env, use_existing=False, stack_prob=0.0):
        """Set objects to random positions"""
        if not use_existing:
            env.reset()  # Clear all objects
            for i in range(self.max_num_obj):
                env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=None,
                    stack_prob=stack_prob,
                )
        else:
            # Using existing objects
            env.move_all_objects_to_buffer()
            for obj_id in env.obj_ids["rigid"]:
                env.move_object_to_random(obj_id, prior=None, stack_prob=stack_prob)