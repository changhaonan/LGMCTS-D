from __future__ import annotations
from typing import Literal
from lgmcts.env.base import BaseEnv
from lgmcts.env.wrappers.prompt_renderer import PromptRenderer
from lgmcts.env.wrappers.recorder import GUIRecorder

from lgmcts.tasks import ALL_TASKS, ALL_PARTITIONS, PARTITION_TO_SPECS


def make(
    task_name: str | None,
    *,
    task_kwargs: dict | None = None,
    modalities: Literal["rgb", "depth", "segm"]
    | list[Literal["rgb", "depth", "segm"]]
    | None = None,
    seed: int | None = None,
    debug: bool = False,
    display_debug_window: bool = False,
    render_prompt: bool = False,
    record_gui: bool = False,
    record_kwargs: dict | None = None,
    hide_arm_rgb: bool = True,
) -> BaseEnv:
    if record_gui:
        record_kwargs = record_kwargs or dict(video_name="gui_record.mp4")
        env = GUIRecorder(
            modalities=modalities,
            task=task_name,
            task_kwargs=task_kwargs,
            seed=seed,
            debug=debug,
            display_debug_window=display_debug_window,
            hide_arm_rgb=hide_arm_rgb,
            **record_kwargs,
        )
    else:
        env = BaseEnv(
            task=task_name,
            modalities=modalities,
            task_kwargs=task_kwargs,
            seed=seed,
            debug=debug,
            display_debug_window=display_debug_window,
            hide_arm_rgb=hide_arm_rgb,
        )
    if render_prompt:
        env = PromptRenderer(env)
    return env