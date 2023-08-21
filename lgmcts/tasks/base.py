from __future__ import annotations
import os
import json
import cv2
import gym
import numpy as np
import random
import importlib_resources
from copy import deepcopy
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