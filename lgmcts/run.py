from __future__ import annotations
import os
import lgmcts
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv


def build_env_and_task(
    task_name: str,
    task_kwargs: dict | None,
    modalities,
    seed: int | None = None,
    debug: bool = False,
):
    env = lgmcts.make(
        task_name=task_name, task_kwargs=task_kwargs, modalities=modalities, seed=seed, debug=debug, display_debug_window=debug,
    )
    task = env.task
    return env, task

if __name__ == '__main__':
    task_name = "structure_rearrange"
    env, task = build_env_and_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm"],
        seed=0,
        debug=True,
    )
    task.reset(env)  # init
    env.move_all_objects_to_buffer()
    for i in range(10):
        task.update_env(env)
        env.step()
        task.update_goals()