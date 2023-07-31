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
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=True,
    )

    obs_cache = []
    for i in range(10):
        # Start-config
        obs = env.reset()
        obs_cache.append(obs)
        elapsed_steps = 0
        meta, prompt, prompt_assets = env.meta_info, env.prompt, env.prompt_assets

        # Set to start state
        obs = task.start(env)
        obs_cache.append(obs)

        task.gen_goal_spec(env)