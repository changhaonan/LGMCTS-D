from __future__ import annotations
import cv2
import lgmcts
import numpy as np
from lgmcts.env import seed


if __name__ == "__main__":
    debug = True
    task_name = f"push_object_{seed}"
    resolution = 0.01
    pix_padding = 1  # padding for clearance
    n_samples = 5
    num_save_digits = 6
    env = lgmcts.make(
        task_name=task_name,
        task_kwargs=lgmcts.PARTITION_TO_SPECS["train"][task_name],
        modalities=["rgb", "segm", "depth"],
        seed=0,
        debug=debug,
        display_debug_window=debug,
        hide_arm_rgb=False,
    )
    env.reset()

    for i in range(100):
        # random action
        pose0_position = np.random.uniform(0.3, 0.5, size=3)
        pose0_rotation = np.array([0, 0, 0, 1])
        pose1_position = np.random.uniform(0.3, 0.5, size=3)
        pose1_rotation = np.array([0, 0, 0, 1])
        action = {
            "pose0_position": pose0_position,
            "pose0_rotation": pose0_rotation,
            "pose1_position": pose1_position,
            "pose1_rotation": pose1_rotation,
        }
        # execute action
        obs, _, terminated, __, info = env.step(action)
        front_rgb = obs["rgb"]["front"].copy().transpose(1, 2, 0)
        front_rgb = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("front_rgb", front_rgb)
        cv2.waitKey(1)
        pass
    env.close()
