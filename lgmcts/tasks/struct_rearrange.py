from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
from copy import deepcopy
import lgmcts.utils.misc_utils as utils
from lgmcts.tasks import BaseTask
from lgmcts.encyclopedia import ObjPedia, TexturePedia
from lgmcts.placeholders import PlaceholderText, PlaceholderObj, PlaceholderScene


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None

    
class StructRearrange(BaseTask):
    task_name = "struct_rearrange"

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
                "types": obj_express_types,
            }
            for i in range(1, max_num_obj + 1)
        } 
        placeholder_expression["pattern"] = { "types": ["text"] }
        # template
        prompt_template = [
            "Set the table to {pattern}",
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
        # temporary data
        self.goal_pattern_info = {}

    def reset(self, env):
        """Reset the scene to goal state"""
        super().reset(env)
        pattern_type = env.rng.choice(self.pattern_types)
        self.set_objects_to_pattern(env, pattern_type, False, self.stack_prob)  # Structured Goal

    def start(self, env):
        """Reset the env to start state"""
        self.set_objects_to_random(env, True, self.stack_prob) # Random Init
        obs, _, _, _, _ = env.step()
        return obs

    def check_success(self):
        if self.progress == 0 or self.progress == 1:
            return ResultTuple(success=False, failure=True, distance=None)
        elif self.progress == 2:
            return ResultTuple(success=True, failure=False, distance=None)
        else:
            raise ValueError("Invalid progress value")

    def set_objects_to_pattern(self, env, pattern_type: str, use_existing=False, stack_prob=0.0):
        """Set objects to a line, use_existing decides whether to add new object or not"""
        pattern_prior, self.goal_pattern_info = utils.gen_random_pattern(pattern_type, env.ws_map_size, env.rng)
        # update goal pattern info
        
        # Add object
        if not use_existing:
            for i in range(self.max_num_obj):
                env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=pattern_prior,
                    stack_prob=stack_prob,
                )
        else:
            raise NotImplementedError("Not implemented yet")
        self.placeholders["pattern"] = PlaceholderText(pattern_type)

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

    def gen_goal_spec(self, env):
        """goal specification; used for StructDiffusion"""
        spec = super().gen_goal_spec(env)
        # anchor object
        spec["anchor"] = {
            "objects": []
        }

        # rearrange object
        spec["rearrange"] = {
            "combine_features_logic": "None",
            "count": "None",
            "objects": []
        }
        for obj_id in env.obj_ids["rigid"]:
            obj_info = env.obj_id_reverse_mapping[obj_id]
            spec["rearrange"]["objects"].append(
                {
                    "obj_id": obj_id,
                    "obj_name": obj_info["obj_name"],
                    "obj_assets": obj_info["obj_assets"],
                }
            )

        # distract object
        spec["distract"] = {
            "objects": []
        }  # Empty for now

        # shape information (pattern)
        # append pattern information
        self.goal_pattern_info["position"] = utils.pix_to_xyz(self.goal_pattern_info["position_pixel"], None, env.bounds, env.pix_size, True)
        if "radius_pixel" in self.goal_pattern_info:
            self.goal_pattern_info["radius"] = self.goal_pattern_info["radius_pixel"] * env.pix_size
        spec["shape"] = self.goal_pattern_info

        # 
        return spec

    def gen_type_vocabs(self):
        type_vocabs = super().gen_type_vocabs()
        # update class according to task
        type_vocabs["class"] = {}
        for i, obj in enumerate(self.obj_list):
            type_vocabs["class"][obj.name] = i
        type_vocabs["color"] = {}
        for i, color in enumerate(self.color_list):
            type_vocabs["color"][color.name] = i
        return type_vocabs