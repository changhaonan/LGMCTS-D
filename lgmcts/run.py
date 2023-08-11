from __future__ import annotations
import os
import lgmcts
from lgmcts import PARTITION_TO_SPECS
import lgmcts.utils.file_utils as U
from lgmcts.tasks import BaseTask
from lgmcts.env.base import BaseEnv
from lgmcts.components.prompt import PromptGenerator
from lgmcts.algorithm import Sampler


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
    task_name = "struct_rearrange"
    env, task = build_env_and_task(
        task_name,
        PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=True,
    )

    obs_cache = []
    prompt_generator = PromptGenerator(env.rng)
    sampler = Sampler()
    for i in range(10):
        # reset
        obs = env.reset()
        sampler.reset()
        prompt_generator.reset()
        # generate goal
        prompt_str, obs = task.gen_goal_config(env, prompt_generator)
        obs = task.gen_start_config(env)
        ## Test a sampling process
        ## Step 1. update the sampler
        sampler.update(obs)
        ## Step 2. sample a goal
        goals = task.goals
        sampler.sample(goal)

        print(f"==== Episode {i} ====")
        print(prompt_str)