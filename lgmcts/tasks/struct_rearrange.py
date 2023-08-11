from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
from copy import deepcopy
import lgmcts.utils.misc_utils as utils
import lgmcts.utils.spatial_utils as spatial_utils
import lgmcts.utils.pybullet_utils as pybullet_utils
from lgmcts.tasks import BaseTask
from lgmcts.components.encyclopedia import ObjPedia, TexturePedia
from lgmcts.components.placeholders import PlaceholderText
from lgmcts.components.prompt import PromptGenerator


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None

    
class StructRearrange(BaseTask):
    """Structured Rearrange Task"""
    task_name = "struct_rearrange"

    def __init__(
        self, 
        # ==== task specific ====
        max_num_obj: int = 12,
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
        # pattern_type = env.rng.choice(self.pattern_types)
        # self.add_objects_to_pattern(env, self.max_num_obj, pattern_type, False, self.stack_prob)  # Structured Goal

    def start(self, env):
        """Reset the env to start state"""
        env.reset()  # Clear all objects
        self.add_objects_to_random(env, self.max_num_obj, True, self.stack_prob) # Random Init
        obs, _, _, _, _ = env.step()
        return obs

    def check_success(self, *args, **kwargs) -> NamedTuple:
        if self.progress == 0 or self.progress == 1:
            return ResultTuple(success=False, failure=True, distance=None)
        elif self.progress == 2:
            return ResultTuple(success=True, failure=False, distance=None)
        else:
            raise ValueError("Invalid progress value")

    def add_objects_to_pattern(self, env, max_num_obj: int, pattern_type: str, use_existing: bool=False, stack_prob: float=0.0):
        """Set objects to a line, use_existing decides whether to add new object or not"""
        pattern_prior, self.goal_pattern_info = utils.gen_random_pattern(pattern_type, env.ws_map_size, env.rng)
        # Add object
        added_obj_ids = []
        if not use_existing:
            for i in range(max_num_obj):
                obj_id, _, __ = env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=pattern_prior,
                    stack_prob=stack_prob,
                )
                if obj_id is not None:
                    added_obj_ids.append(obj_id)
        else:
            raise NotImplementedError("Not implemented yet")
        self.placeholders["pattern"] = PlaceholderText(pattern_type)
        return added_obj_ids

    def add_objects_to_random(self, env, max_num_obj: int, use_existing: bool=False, stack_prob :float=0.0):
        """Set objects to random positions"""
        added_obj_ids = []
        if not use_existing:
            for i in range(max_num_obj):
                obj_id, _, __ = env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=None,
                    stack_prob=stack_prob,
                )
                if obj_id is not None:
                    added_obj_ids.append(obj_id)
        else:
            # Using existing objects
            env.move_all_objects_to_buffer()
            for obj_id in env.obj_ids["rigid"]:
                env.move_object_to_random(obj_id, prior=None, stack_prob=stack_prob)
                added_obj_ids.append(obj_id)
        return added_obj_ids

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

    def gen_goal_config(self, env, prompt: PromptGenerator):
        """Generate goal config"""
        #TODO: add region prompt later, currently not supported
        ## Step 1: generate a random pattern
        pattern_type = env.rng.choice(self.pattern_types)
        max_num_pattern = int(self.max_num_obj/2)
        pattern_obj_ids = self.add_objects_to_pattern(env, max_num_pattern, pattern_type, False, self.stack_prob)
        # parse object names
        pattern_obj_names = [f"{env.obj_id_reverse_mapping[obj_id]['texture_name']} {env.obj_id_reverse_mapping[obj_id]['obj_name']}" for obj_id in pattern_obj_ids]
        prompt.gen_pattern_prompt(pattern_obj_names, pattern_type)
        ## Step 2: add some more objects & spatial relationship
        max_num_add = int(self.max_num_obj/2)
        added_obj_ids = self.add_objects_to_random(env, max_num_add, False, self.stack_prob)
        # randomly select one from pattern obj and added obj
        pair_obj_ids = env.rng.choice(pattern_obj_ids + added_obj_ids, 2)
        pair_obj_names = [f"{env.obj_id_reverse_mapping[obj_id]['texture_name']} {env.obj_id_reverse_mapping[obj_id]['obj_name']}" for obj_id in pair_obj_ids]
        # compute spatial from the pair
        aabb_1 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[0])
        aabb_2 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[1])
        pose_1 = spatial_utils.Points9.from_aabb(aabb_1[0], aabb_1[1])
        pose_2 = spatial_utils.Points9.from_aabb(aabb_2[0], aabb_2[1])
        spatial_label = spatial_utils.Points9.label(pose_1, pose_2)
        spatial_str_list = spatial_utils.Points9.vocabulary(spatial_label)
        if spatial_str_list[0] != "A has no relationship with B":
            spatial_rel = self.rng.choice(spatial_str_list)
            prompt.gen_pair_prompt(pair_obj_names[0], pair_obj_names[1], spatial_rel[4:-1].strip())
        
        # Env step forward
        obs, _, _, _, _ = env.step()
        return prompt.prompt, obs

    def gen_start_config(self, env):
        """Generate a random config using existing objects"""
        self.add_objects_to_random(env, self.max_num_obj, True, self.stack_prob)

        # Env step forward
        obs, _, _, _, _ = env.step()
        return obs
