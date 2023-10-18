from __future__ import annotations
import os
import json
import cv2
import gym
import numpy as np
import random
import importlib_resources
from copy import deepcopy
import warnings
from typing import Literal, NamedTuple
from lgmcts.components.end_effectors import Suction
from lgmcts.components.action_primitives import PickPlace


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None


class BaseTask:
    task_name: str
    REJECT_SAMPLING_MAX_TIMES = 10

    def __init__(
        self,
        modalities: list[str],
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        self.assets_root = None
        self.goals = []
        self.progress = 0
        self.placeholders = {}
        self.seed = seed
        self.set_seed(seed)
        self.ee = Suction  # ee is bined to the task
        self.primitive = PickPlace()  # primitive is bined to the task
        self.prompt = ""
    
    def reset(self, env):
        self.client_id = env.client_id

        if not self.assets_root:
            raise ValueError("assets_root must be set")
        self.goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self.placeholders = {}

    def set_difficulty(self, difficulty: Literal["easy", "medium", "hard"]):
        self.difficulty_level = difficulty

    def update_goals(
        self, env
    ): 
        pass

    def update_env(self, env):
        """Update environment according to env progress"""
        return 

    def check_success(self, *args, **kwargs) -> ResultTuple:
        """
        Check success. It should return a tuple of two boolean values, (success, failure).
        A trajectory will be terminated if fails.
        This function may be invoked at the final step.
        It may also be invoked every step for "constraint satisfaction" tasks.
        """
        raise NotImplementedError

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

    def oracle_action(self, env):
        """Get oracle action Given current states"""
        raise NotImplementedError

    ######## Helper functions ########
    def add_objects_to_pattern(
            self, env, objs, colors, pattern_prior: np.ndarray | None, num_limit: list[int] = [0, 100], use_existing: bool = False, stack_prob: float = 0.0):
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
                    if len(added_obj_ids) >= num_limit[-1]:
                        break
        else:
            raise NotImplementedError("Not implemented yet")
        if len(added_obj_ids) == 0:
            if pattern_prior is not None:
                warnings.warn("No object is added to the pattern")
            # assert False, "No object is added to the pattern"
        return added_obj_ids, obj_status

    def add_objects_to_random(self, env, max_num_obj: int, obj_candidates: list = [], color_candidates: list = [], use_existing: bool = False, stack_prob: float = 0.0):
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
                    print("Could not add object with id: ", obj_id)
        else:
            # Using existing objects
            env.move_all_objects_to_buffer()
            for obj_id in env.obj_ids["rigid"]:
                env.move_object_to_random(obj_id, prior=None, stack_prob=stack_prob)
                added_obj_ids.append(obj_id)
        return added_obj_ids

    # Struct-diffusion style goal specification
    def gen_goal_spec(self, env):
        spec = {}
        return spec

    def gen_type_vocabs(self):
        # load the template
        template_path = os.path.join(self.assets_root, "templates", "type_vocabs_coarse.json")
        with open(template_path, "r") as f:
            type_vocabs = json.load(f)
        return type_vocabs

    ## Restore & save function
    def get_state(self):
        state = {
            "goals": self.goals,
            "progress": self.progress,
            "prompt": self.prompt,
            "seed": self.seed,
        }
        return state
    
    def set_state(self, state):
        self.goals = state["goals"]
        self.progress = state["progress"]
        self.prompt = state["prompt"]
        self.set_seed(state["seed"])