from __future__ import annotations
import cv2
import gym
import importlib_resources
from typing import Literal, NamedTuple


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None


class BaseTask:
    task_name: str
    REJECT_SAMPLING_MAX_TIMES = 10

    def __init__(
        self,
        prompt_template: str,
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
    
    def reset(self, env):
        env.reset()
        self.client_id = env.client_id

        if not self.assets_root:
            raise ValueError("assets_root must be set")
        self.goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self.placeholders = {}

    def set_difficulty(self, difficulty: Literal["easy", "medium", "hard"]):
        self.difficulty_level = difficulty

    def update_goals(
        self
    ): 
        pass

    def update_env(self, env):
        """Update environment according to env progress"""
        return 

    def check_success(self):
        return ResultTuple(success=True, failure=False, distance=None)

    def generate_prompt(self, *args, **kwargs):
        return "", []