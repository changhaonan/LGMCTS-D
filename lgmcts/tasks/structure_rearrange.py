from __future__ import annotations
from typing import Literal, NamedTuple
import cv2
import lgmcts.utils.misc_utils as utils
from lgmcts.tasks import BaseTask
from lgmcts.encyclopedia import ObjPedia, TexturePedia


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None

    
class StructureRearrange(BaseTask):
    task_name = "structure_rearrange"

    def __init__(
        self, 
        # ==== task specific ====
        num_object: int = 4,
        obj_list: list[str] | None = None,
        color_list: list[str] | None = None,
        # ==== general ====
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,):
        super().__init__(
            prompt_template="Rearrange to this {structure}",
            modalities=["rgb"],
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            seed=seed,
            debug=debug,
        )
        self.num_object = num_object
        self.obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in obj_list]
        self.color_list = [TexturePedia.lookup_color_by_name(color) for color in color_list]
        # 
        self.obs_img_size = obs_img_size

    def update_goals(self):
        # There is two stages in this task: start and end
        self.progress += 1
        if self.progress >= 2:
            self.progress = 0

    def update_env(self, env):
        if self.progress == 0:
            self.set_objects_to_line(env, False)  # Add new objects into a line
        elif self.progress == 1:
            self.set_objects_to_random(env, True) # Put existing objects to random positions
        
        env.wait_until_settle()

    def check_success(self):
        if self.progress == 0:
            return ResultTuple(success=True, failure=False, distance=None)
        elif self.progress == 1:
            return ResultTuple(success=False, failure=True, distance=None)
        else:
            raise ValueError("Invalid progress value")

    def set_objects_to_line(self, env, use_existing=False):
        """Set objects to a line, use_existing decides whether to add new object or not"""
        line_pattern = utils.gen_random_pattern("line", env.occupy_size, env.rng)
        # Add object
        if not use_existing:
            env.reset()  # Clear all objects
            for i in range(self.num_object):
                env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=line_pattern,
                )

    def set_objects_to_random(self, env, use_existing=False):
        """Set objects to random positions"""
        if not use_existing:
            env.reset()  # Clear all objects
            for i in range(self.num_object):
                env.add_random_object_to_env(
                    obj_lists=self.obj_list,
                    color_lists=self.color_list,
                    prior=None,
                )
        else:
            # Collect all objects info
            obj_infos = env.obj_id_reverse_mapping
            if len(obj_infos) != 0:
                env.reset()  # Clear all objects
                for obj_id, obj_info in obj_infos:
                    obj_entry = None
                    texture_entry = None
                    obj_size = [1.0, 1.0, 1.0]
                    env.add_object_to_env(
                        obj_entry,
                        texture_entry,
                        obj_size,
                        category="rigid"
                    )