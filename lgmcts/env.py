from __future__ import annotations
import numpy as np
import tempfile
import pybullet as p
import pybullet_data
import threading
import time
import os
import cv2
import math
import traceback
from typing import Dict, Any, Tuple, Union, List
import os
from PIL import Image
import pybullet_utils
from cameras import get_agent_cam_config
from encyclopedia import ObjPedia, TexturePedia, ObjEntry, TextureEntry

UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"
UR5_URDF_PATH = "ur5/ur5.urdf"

class TableSceneBase:
    """A simple table top scene"""
    def __init__(
        self, 
        assets_root: str, 
        modalities,
        obs_img_size: Tuple[int, int] = (128, 256),
        obs_img_views: List[str] = ["front", "top"],
        seed:int = 0,
        hz: int = 240,
        max_sim_steps_to_static: int = 1000,
        debug: bool = False,
        display_debug_window: bool = False, 
        hide_arm_rgb: bool = False,
        ):
        self.assets_root = assets_root
        self.obj_ids = {"fixed": [], "rigid": []}
        self.add_object_id_reverse_mapping_info = {}
        # obj_id_reverse_mapping: a reverse mapping dict that maps object unique id to:
        # 1. object_name appended with color name
        # 2. object_texture entry in TexturePedia
        # 3. object_description entry in ObjPedia

        # Configure pybullet
        self.dt = 1 / 480
        self.sim_step = 0

        # setup modalities
        modalities = modalities or ["rgb", "segm"]
        if isinstance(modalities, str):
            modalities = [modalities]
        assert set(modalities).issubset(
            {"rgb", "depth", "segm"}
        ), f"Unsupported modalities provided {modalities}"
        assert "depth" not in modalities, "FIXME: fix depth normalization"
        self.modalities = modalities

        # setup camera
        self.obs_img_size = obs_img_size
        self.obs_img_views = obs_img_views
        self.set_up_camera(obs_img_size, obs_img_views)

        # Start PyBullet.
        client = self.connect_pybullet_hook(display_debug_window)
        self.client_id = client
        self.client_id = client
        file_io = p.loadPlugin("fileIOPlugin", physicsClientId=client)
        if file_io < 0:
            raise RuntimeError("pybullet: cannot load FileIO!")
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client,
            )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(assets_root, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)
        p.setTimeStep(1.0 / hz, physicsClientId=self.client_id)

        # If display debug window, move default camera closer to the scene.
        if display_debug_window:
            target = p.getDebugVisualizerCamera(physicsClientId=self.client_id)[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,
                physicsClientId=self.client_id,
            )

        assert max_sim_steps_to_static > 0
        self._max_sim_steps_to_static = max_sim_steps_to_static

        self.seed(seed)
        self._display_debug_window = display_debug_window
        self._hide_arm_rgb = hide_arm_rgb

    def connect_pybullet_hook(self, display_debug_window: bool):
        return p.connect(p.DIRECT if not display_debug_window else p.GUI)

    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.add_object_id_reverse_mapping_info = {}

        self.step_counter = 0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        # Temporarily disable rendering to load scene faster.
        if self._display_debug_window:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
            )

        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, PLANE_URDF_PATH),
            [0, 0, -0.001],
            physicsClientId=self.client_id,
        )

        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
            [0.5, 0, 0],
            physicsClientId=self.client_id,
        )

        self.ur5 = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, UR5_URDF_PATH),
            physicsClientId=self.client_id,
        )
        if self._hide_arm_rgb:
            pybullet_utils.set_visibility_bullet(
                self.client_id, self.ur5, pybullet_utils.INVISIBLE_ALPHA
            )
        
        # Re-enable rendering.
        if self._display_debug_window:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )

    def step(self):
        # Step simulator asynchronously until objects settle.
        counter = 0
        while not self.is_static:
            self.step_simulation()
            if counter > self._max_sim_steps_to_static:
                print(
                    f"WARNING: step until static exceeds max {self._max_sim_steps_to_static} steps!"
                )
                break
            counter += 1
    
    def step_simulation(self):
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_counter += 1
    
    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(p.getBaseVelocity(i, physicsClientId=self.client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    # Gym functions
    def close(self):
        p.disconnect(physicsClientId=self.client_id)

    def seed(self, seed=None):
        self._random = np.random.default_rng(seed=seed)
        self._env_seed = seed
        return seed

    # Render & obs
    def set_up_camera(self, obs_img_size: tuple[int, int], obs_img_views: list[str]):
        obs_img_views = obs_img_views or ["front", "top"]
        all_cam_config = get_agent_cam_config(obs_img_size)
        self.agent_cam_config = {view: all_cam_config[view] for view in obs_img_views}

    def render_camera(self, config, image_size=None):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config["image_size"]

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(
            config["rotation"], physicsClientId=self.client_id
        )
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(
            config["position"], lookat, updir, physicsClientId=self.client_id
        )
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar, physicsClientId=self.client_id
        )

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))
        # transpose from HWC to CHW
        color = np.transpose(color, (2, 0, 1))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)
        # normalize depth to be within range [0, 1]
        depth /= 20.0
        # add 'C' dimension
        depth = depth[np.newaxis, ...]

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def _get_obs(self):
        obs = {f"{modality}": {} for modality in self.modalities}

        for view, config in self.agent_cam_config.items():
            color, depth, segm = self.render_camera(config)
            render_result = {"rgb": color, "depth": depth, "segm": segm}
            for modality in self.modalities:
                obs[modality][view] = render_result[modality]

        # assert self.observation_space.contains(obs)
        return obs

    # Add objects related
    def add_object_to_env(
        self,
        env,
        obj_entry: ObjEntry,
        color: TextureEntry,
        size: tuple[float, float, float],
        scalar: float | list[float] = 1.0,
        pose: tuple[tuple, tuple] = None,
        category: str = "rigid",
        retain_temp: bool = True,
        **kwargs,
    ):
        """helper function for adding object to env."""
        scaled_size = self._scale_size(size, scalar)
        if pose is None:
            pose = self.get_random_pose(env, scaled_size)
        if pose[0] is None or pose[1] is None:
            # reject sample because of no extra space to use (obj type & size) sampled outside this helper function
            return None, None, None
        obj_id, urdf_full_path = pybullet_utils.add_any_object(
            env=env,
            obj_entry=obj_entry,
            pose=pose,
            size=scaled_size,
            scaling=scalar,
            retain_temp=retain_temp,
            category=category,
            **kwargs,
        )
        if obj_id is None:  # pybullet loaded error.
            return None, urdf_full_path, pose
        # change texture
        pybullet_utils.p_change_texture(obj_id, color, env.client_id)
        # add mapping info
        pybullet_utils.add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=obj_entry,
            texture_entry=color,
        )

        return obj_id, urdf_full_path, pose

    def add_random_object_to_env(
        self, 
        env,
        obj_lists: list[ObjEntry],
        color_lists: list[TextureEntry],
        **kwargs,
    ):
        """Add random an object from list, with a random texture and random size"""
        sampled_obj = self.rng.choice(obj_lists).value
        sampled_obj_size = self.rng.uniform(
            low=sampled_obj.size_range.low,
            high=sampled_obj.size_range.high,
        )
        if len(color_lists) > 1:
            sampled_obj_color = self.rng.choice(color_lists).value
        elif len(color_lists) == 1:
            sampled_obj_color = color_lists[0].value
        else:
            sampled_obj_color = None
        
        obj_id, urdf, pose = self.add_object_to_env(
            env,
            sampled_obj,
            sampled_obj_color,
            sampled_obj_size,
            category="rigid",
        )

if __name__ == '__main__':
    assets_root = os.path.join(os.path.dirname(__file__), 'assets')
    scene = TableSceneBase(assets_root=assets_root, modalities=['rgb'], display_debug_window=True)
    scene.reset()
    
    for i in range(1000):
        scene.step_simulation()
    # Check
    # color = obs['rgb']['top'].transpose(1, 2, 0)
    # cv2.imshow('color', color)
    # cv2.waitKey(0)

    # Test object adding 
    obj_lists = [ObjPedia.BOWL, ObjPedia.BLOCK, ObjPedia.CAPITAL_LETTER_A]
    color_lists = [TexturePedia.RED, TexturePedia.GREEN, TexturePedia.BLUE]
    for i in range(2):
        scene.add_random_object_to_env(scene, obj_lists, color_lists)
        for i in range(1000):
            scene.step_simulation()

    scene.close()
