from __future__ import annotations
import cv2
import gym
import importlib_resources
from copy import deepcopy
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
        placeholder_expression: dict[str, dict[str, str]],
        modalities: list[str],
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        self.prompt_template = prompt_template
        self.placeholder_expression = placeholder_expression
        self.assets_root = None
        self.goals = []
        self.progress = 0
        self.placeholders = {}
        self.seed = seed
    
    def reset(self, env):
        self.client_id = env.client_id

        if not self.assets_root:
            raise ValueError("assets_root must be set")
        self.goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self.placeholders = {}
        return env.reset()

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
        """
        Generate prompt from `self.prompt_template`, 'self.task_meta', and `self.placeholders`.
        This method may be invoked in `env.reset()`.
        Implementation of this method may vary in different tasks.
        """
        expressions = {}
        # for each placeholder items, generate required expressions
        for name, placeholder in self.placeholders.items():
            args = self.placeholder_expression[name]
            expressions[name] = placeholder.get_expression(**args)
        # now assemble the prompt
        prompt = deepcopy(self.prompt_template)
        assets = {}
        for name in self.placeholders:
            replacement = ""
            for expression_type in self.placeholder_expression[name]["types"]:
                if expression_type == "image":
                    replacement = replacement + "{" + name + "} "
                    assets[name] = expressions[name]["image"]
                else:
                    # text expression, e.g., name, novel_name, alias, etc
                    replacement = replacement + expressions[name][expression_type] + " "
            # stripe the last white space
            replacement = replacement[:-1]
            prompt = prompt.replace("{" + f"{name}" + "}", replacement)
        return prompt, assets