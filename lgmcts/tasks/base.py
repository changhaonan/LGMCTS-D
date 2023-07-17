from __future__ import annotations
import cv2
import gym
from lgmcts.env.base import EnvBase
import importlib_resources


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
        # Create scene
        with importlib_resources.files("lgmcts.assets") as p:
            self.scene = EnvBase(
                assets_root=str(p),
                modalities=modalities,
                obs_img_size=obs_img_size,
                obs_img_views=obs_img_views,
                seed=seed,
                debug=debug,
                display_debug_window=debug,
            )

    def reset(self):
        self.scene.reset()