from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
import numpy as np
import warnings
from copy import deepcopy
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
        max_num_obj: int = 10,
        stack_prob: float = 0.0,
        pattern_types: list[str] = ["line", "circle"],
        enable_multi_prompt: bool = False,
        obj_list: list[str] | None = None,
        color_list: list[str] | None = None,
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
        self.max_num_obj = max_num_obj
        self.stack_prob = stack_prob
        self.pattern_types = pattern_types
        self.enable_multi_prompt = enable_multi_prompt
        self.obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in obj_list]
        self.color_list = [TexturePedia.lookup_color_by_name(color) for color in color_list]
        # 
        self.obs_img_size = obs_img_size
        # temporary data
        self.goal_pattern_info = {}
        # for record
        self.distract_obj_ids = []

    def reset(self, env):
        """Reset the task"""
        super().reset(env)
        # pattern_type = env.rng.choice(self.pattern_types)
        # self.add_objects_to_pattern(env, self.max_num_obj, pattern_type, False, self.stack_prob)  # Structured Goal
        self.distract_obj_ids = []

    def add_objects_to_pattern(
        self, env, objs, colors, pattern_prior: np.ndarray | None, use_existing: bool=False, stack_prob: float=0.0):
        """Set objects to a line, use_existing decides whether to add new object or not"""
        # Add object
        added_obj_ids = []
        obj_status = {}
        if not use_existing:
            for obj, color in zip(objs, colors):
                obj_id, _, __ = env.add_random_object_to_env(
                    obj_lists=[obj],
                    color_lists=[color],
                    prior=pattern_prior,
                    stack_prob=stack_prob,
                )
                if obj_id is not None:
                    added_obj_ids.append(obj_id)
                    obj_status[obj_id] = True
        else:
            raise NotImplementedError("Not implemented yet")
        if len(added_obj_ids) == 0:
            if pattern_prior is not None:
                cv2.imshow("prior", pattern_prior)
                cv2.waitKey(0)
            # assert False, "No object is added to the pattern"
        return added_obj_ids, obj_status

    def add_objects_to_random(self, env, max_num_obj: int, obj_candidates: list=[], color_candidates: list=[], use_existing: bool=False, stack_prob :float=0.0):
        """Set objects to random positions
        Args:
            max_num_obj: maximum number of objects to add
            obj_candidates: list of object candidates
            color_candidates: list of color candidates
            use_existing: whether to use existing objects, meaning moving existing objects to random positions
            stack_prob: probability of stacking objects
        """
        obj_list = obj_candidates if len(obj_candidates) > 0 else self.obj_list
        color_list = color_candidates if len(color_candidates) > 0 else self.color_list
        added_obj_ids = []
        if not use_existing:
            for i in range(max_num_obj):
                obj_id, _, __ = env.add_random_object_to_env(
                    obj_lists=obj_list,
                    color_lists=color_list,
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
        #FIXME: what does these contents mean?
        spec = super().gen_goal_spec(env)
        goal = self.goals[0]
        # anchor object
        spec["anchor"] = {
            "objects": []
        }
        spec["anchor"]["objects"].append(
            {
                "obj_id": goal["anchor_id"],
                "obj_name": env.obj_id_reverse_mapping[goal["anchor_id"]]["obj_name"],
                "obj_assets": env.obj_id_reverse_mapping[goal["anchor_id"]]["obj_assets"],
            }
        )
        # rearrange object
        spec["rearrange"] = {
            "combine_features_logic": "None",
            "count": "None",
            "objects": []
        }
        for obj_id in goal["obj_ids"]:
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
        }
        for obj_id in self.distract_obj_ids:
            obj_info = env.obj_id_reverse_mapping[obj_id]
            spec["distract"]["objects"].append(
                {
                    "obj_id": obj_id,
                    "obj_name": obj_info["obj_name"],
                    "obj_assets": obj_info["obj_assets"],
                }
            )

        # shape information (pattern)
        # append pattern information
        goal = self.goals[0]  # FIXME: currently we only support one goal for struct_diffusion
        shape_info = {}
        shape_info["position"] = utils.pix_to_xyz(goal["position_pixel"][:2], None, env.bounds, env.pix_size, True)
        if "radius_pixel" in goal:
            shape_info["radius"] = goal["radius_pixel"] * env.pix_size
        spec["shape"] = shape_info

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
        print("type_vocabs:", type_vocabs)
        return type_vocabs

    def gen_goal_config(self, env, promptor: PromptGenerator, obj_selector: ObjectSelector, **kwargs):
        """Generate goal config"""
        num_color = kwargs.get("num_color", 2)  # Each scene only has X colors
        num_patterns = 2 if self.enable_multi_prompt else 1
        num_added_objs = 0

        obj_list = self.rng.choice(self.obj_list, min(self.max_num_obj, len(self.obj_list)), replace=False)   # current candidate
        color_list = self.rng.choice(self.color_list, num_color, replace=False)
        ## Step 1: select object candidates
        for i in range(num_patterns):
            if obj_list is None or len(obj_list) <= 2:
                break  # no more enough candidate to formulate pattern
            selected_objs = obj_list
            selected_colors = self.rng.choice(color_list, len(obj_list), replace=True)
            obj_selector.reset()
            obj_selector.set_objs(selected_objs, selected_colors)
            selection = obj_selector.gen_anchor_obj_prompt()
            if not selection:  # no valid selection
                continue
            ## Step 2: select pattern & add objects to scene
            if selection["anchor_obj"] is not None:
                [anchor_id], _ = self.add_objects_to_pattern(env, [selection["anchor_obj"]], [selection["anchor_color"]], None, False, 0.0)  # add anchor object
            else:
                anchor_id = -1
            # generate pattern
            pattern_type = env.rng.choice(self.pattern_types)
            max_try = 3
            rearrange_obj_ids = []
            pattern_info = {}
            for i in range(max_try):
                try:
                    pattern_prior, pattern_info = PATTERN_DICT[pattern_type].gen_prior(env.ws_map_size, env.rng)
                    rearrange_obj_ids, obj_status = self.add_objects_to_pattern(env, selection["in_obj"], selection["in_color"], pattern_prior, False, 0.0)
                    assert len(rearrange_obj_ids) > 0, "No object is added to the pattern"
                    break
                except:
                    continue
            if anchor_id == -1:
                anchor_id = rearrange_obj_ids[0]
            # update goals
            pattern_info["obj_ids"] = rearrange_obj_ids
            pattern_info["anchor_id"] = anchor_id
            self.goals.append(pattern_info)
            # update prompt
            promptor.gen_pattern_prompt(selection["prompt_str"], pattern_type)
            # update obj
            num_added_objs += len(rearrange_obj_ids)
            obj_list = selection["out_obj"]
            color_list = selection["out_color"]
            ##DEBUG
            if i == 0:
                break

        ## Step 3: add distract objects
        num_distract = self.max_num_obj - num_added_objs - 1
        if num_distract > 0 and len(selection["out_obj"]) > 0:  # exists unselected
            self.distract_obj_ids = self.add_objects_to_random(env, num_distract, selection["out_obj"], selection["out_color"], False, 0.0)
        else:
            self.distract_obj_ids = []
        num_distract = len(self.distract_obj_ids)
        ## Step 4: 
        if self.enable_multi_prompt:
            if len(self.goals) < num_patterns and num_distract > 0:  # not enough pattern
                # 4.1 add a spatial prompt
                # randomly select one from pattern obj and added obj
                anchor_id = env.rng.choice(rearrange_obj_ids)
                place_id = env.rng.choice(self.distract_obj_ids)
                pair_obj_ids = [anchor_id, place_id]
                pair_obj_names = [f"{env.obj_id_reverse_mapping[obj_id]['texture_name']} {env.obj_id_reverse_mapping[obj_id]['obj_name']}" for obj_id in pair_obj_ids]
                # compute spatial from the pair
                aabb_1 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[0])
                aabb_2 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[1])
                pose_1 = spatial_utils.Points9.from_aabb(aabb_1[0], aabb_1[1])
                pose_2 = spatial_utils.Points9.from_aabb(aabb_2[0], aabb_2[1])
                spatial_label = spatial_utils.Points9.label(pose_2, pose_1)
                spatial_str_list = spatial_utils.Points9.vocabulary(spatial_label)
                # FIXME: orignal spatial_label has 8 dim, but in LGMCTS, we only use 4 dim
                spatial_label = spatial_label[:4]
                if spatial_str_list[0] != "A has no relationship with B":
                    spatial_rel = self.rng.choice(spatial_str_list)
                    promptor.gen_pair_prompt(pair_obj_names[1], pair_obj_names[0], spatial_rel[4:-1].strip())
                    # update goal
                    self.goals.append(
                        {
                            "type": "pattern:spatial",
                            "obj_ids": pair_obj_ids,
                            "spatial_label": spatial_label,
                            "spatial_str": spatial_rel
                        }
                    )      

        ## Step 5: assemble prompt and goal specific
        # gen prompt
        promptor.gen_prompt()
        self.prompt = promptor.prompt
        # Env step forward
        obs, _, _, _, _ = env.step()
        return self.prompt, obs

    # def gen_goal_config(self, env, promptor: PromptGenerator, obj_selector: ObjectSelector) -> tuple[str, dict]:
    #     """Generate goal config. The goal of object selector is selection a set of object, 
    #     with some specific pattern.
    #     """
    #     #TODO: add region prompt later, currently not supported
    #     max_try = 3  # max try for sampling
    #     obj_selector.reset()
    #     obj_selector.set_objs(env.obj_ids["rigid"], env.obj_list, env.color_list)

    #     # Step 1: select objects & colors
    #     num_color = 2  # Each scene only has X colors
    #     num_object = 6  # Each scene only has at most X objects
    #     selected_objs = self.rng.choice(env.obj_list, num_object, replace=False)
    #     color_candidates = self.rng.choice(env.color_list, num_color, replace=False)
    #     selected_colors = self.rng.choice(color_candidates, num_object, replace=True)

    #     ## Step 2: generate a random pattern & add objects to the scene
    #     pattern_type = env.rng.choice(self.pattern_types)
    #     # max_num_pattern = int(self.max_num_obj/2)
    #     max_num_pattern = 3
    #     for i in range(max_try):
    #         try:
    #             pattern_prior, pattern_info = PATTERN_DICT[pattern_type].gen_prior(env.ws_map_size, env.rng)
    #             pattern_obj_ids = self.add_objects_to_pattern(env, selected_objs, selected_colors, pattern_prior, False, self.stack_prob)
    #             assert len(pattern_obj_ids) > 0, "No object is added to the pattern"
    #             break
    #         except:
    #             continue
        
    #     promptor.gen_pattern_prompt(obj_str, pattern_type)
    #     # update goals
    #     pattern_info["obj_ids"] = pattern_obj_ids
    #     self.goals.append(pattern_info)

    #     ## Step 2: add some more objects & spatial relationship
    #     # max_num_add = int(self.max_num_obj/4)
    #     # max_num_add = 1  #FIXME: only add one object for now
    #     # added_obj_ids = self.add_objects_to_random(env, max_num_add, False, self.stack_prob)
    #     # # randomly select one from pattern obj and added obj
    #     # pair_obj_ids = env.rng.choice(pattern_obj_ids + added_obj_ids, 2)
    #     # pair_obj_names = [f"{env.obj_id_reverse_mapping[obj_id]['texture_name']} {env.obj_id_reverse_mapping[obj_id]['obj_name']}" for obj_id in pair_obj_ids]
    #     # # compute spatial from the pair
    #     # aabb_1 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[0])
    #     # aabb_2 = pybullet_utils.get_obj_aabb(env, pair_obj_ids[1])
    #     # pose_1 = spatial_utils.Points9.from_aabb(aabb_1[0], aabb_1[1])
    #     # pose_2 = spatial_utils.Points9.from_aabb(aabb_2[0], aabb_2[1])
    #     # spatial_label = spatial_utils.Points9.label(pose_1, pose_2)
    #     # spatial_str_list = spatial_utils.Points9.vocabulary(spatial_label)
    #     # if spatial_str_list[0] != "A has no relationship with B":
    #     #     spatial_rel = self.rng.choice(spatial_str_list)
    #     #     prompt.gen_pair_prompt(pair_obj_names[0], pair_obj_names[1], spatial_rel[4:-1].strip())
    #     #     # update goal
    #     #     self.goals.append(
    #     #         {
    #     #         "type": "pattern:spatial",
    #     #         "obj_ids": pair_obj_ids,
    #     #         "spatial_label": spatial_label,
    #     #         "spatial_str": spatial_rel
    #     #         }
    #     #     )
    #     # Env step forward
    #     obs, _, _, _, _ = env.step()
    #     #
    #     self.prompt = promptor.prompt
    #     return self.prompt, obs

    def gen_start_config(self, env) -> dict:
        """Generate a random config using existing objects"""
        self.add_objects_to_random(env, self.max_num_obj, use_existing=True, stack_prob=self.stack_prob)

        # Env step forward
        obs, _, _, _, _ = env.step()
        return obs

    def check_success(self, *args, **kwargs) -> ResultTuple:
        """Implementation of checking success"""
        if "obj_poses" not in kwargs:
            return ResultTuple(success=False, failure=True, distance=None)
        else:
            obj_poses = kwargs["obj_poses"]
            for goal in self.goals:
                pattern_type = goal["type"].split(":")[-1]
                if pattern_type in PATTERN_DICT:
                    if not PATTERN_DICT[pattern_type].check(obj_poses, pattern_info=goal):
                        return ResultTuple(success=False, failure=True, distance=None)
                else:
                    warnings.warn(f"Pattern type {pattern_type} is not supported")
            return ResultTuple(success=True, failure=False, distance=None)