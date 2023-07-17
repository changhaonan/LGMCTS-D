from __future__ import annotations
from typing import Literal, NamedTuple
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
        self.obj_list = [ObjPedia.lookup_object_by_name(obj) for obj in obj_list]
        self.color_list = [TexturePedia.lookup_color_by_name(color) for color in color_list]
    
    def reset(self, env):
        super().reset(env)

        # Add object
        num_object = 8
        for i in range(num_object):
            env.add_random_object_to_env(
                obj_lists=self.obj_list,
                color_lists=self.color_list,
            )
        # return observation
        obs, _, _, _, _ = env.step()

        return obs

    def update_goals(self):
        # There is two stages in this task: start and end
        self.progress += 1
        if self.progress >= 2:
            self.progress = 0

    def update_env(self, env):
        pass

    def check_success(self):
        if self.progress == 0:
            return ResultTuple(success=True, failure=False, distance=None)
        elif self.progress == 1:
            return ResultTuple(success=False, failure=True, distance=None)
        else:
            raise ValueError("Invalid progress value")