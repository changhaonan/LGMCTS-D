"""Test environment saving and loading"""
from __future__ import annotations
import os
import lgmcts
from lgmcts.components.prompt import PromptGenerator


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
    # init
    task_name = "struct_rearrange"
    resolution = 0.01
    n_samples = 1
    n_generate = 10
    save_path = "/Users/haonanchang/Projects/LGMCTS-D/output/test/save_load"
        
    env, task = build_env_and_task(
        task_name,
        lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=True,
    )
    prompt_generator = PromptGenerator(env.rng)
    print("Generate dataset...")
    for i in range(n_generate):
        # reset
        obs = env.reset()
        prompt_generator.reset()

        # generate goal
        prompt_str, obs = task.gen_goal_config(env, prompt_generator)
        obs = task.gen_start_config(env)
        
        env.save_checkpoint(os.path.join(save_path, f"checkpoint_{i}.pkl"))
        print(f"==== {i} ====")
        print(task.prompt)

    print("Load dataset...")
    for i in range(n_generate):
        print(f"==== {i} ====")
        env.load_checkpoint(os.path.join(save_path, f"checkpoint_{i}.pkl"))
        # Run test here
        obj_poses = env.get_obj_poses()
        result = task.check_success(obj_poses)
        print(f"Epoch {i}: {result.success}")