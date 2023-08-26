from __future__ import annotations
import numpy as np
import tempfile
import pickle
import importlib_resources
import pybullet as p
import pybullet_data
import threading
import time
import os
import cv2
import copy
import open3d as o3d
import math
import traceback
from typing import Dict, Any, Tuple, Union, List, Literal
import os
import gym
from matplotlib import pyplot as plt
from PIL import Image
from anytree import Node, RenderTree
from lgmcts.tasks.base import BaseTask
from lgmcts.tasks import ALL_TASKS as _ALL_TASKS
import lgmcts.utils.pybullet_utils as pybullet_utils
import lgmcts.utils.misc_utils as misc_utils
from lgmcts.components.cameras import get_agent_cam_config, Oracle
from lgmcts.components.encyclopedia import ObjPedia, TexturePedia, ObjEntry, TextureEntry
from lgmcts.components.end_effectors import Suction, Spatula

UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"
UR5_URDF_PATH = "ur5/ur5.urdf"


class BaseEnv:
    """A simple table top scene"""

    def __init__(
        self,
        modalities: Literal["rgb", "depth", "segm"] | list[Literal["rgb", "depth", "segm"]] | None = None,
        task: BaseTask | str | None = None,
        task_kwargs: dict | None = None,
        obs_img_size: Tuple[int, int] = (256, 256),  # (height, width), (128, 256) or (256, 256)
        obs_img_views: List[str] = ["front", "top"],
        seed:int = 0,
        hz: int = 240,
        max_sim_steps_to_static: int = 1000,
        debug: bool = False,
        display_debug_window: bool = False, 
        hide_arm_rgb: bool = False,
        ):
        with importlib_resources.files("lgmcts.assets") as _path:
            self.assets_root = str(_path)
        # Obj infos
        self.obj_ids = {"fixed": [], "rigid": []}
        self.obj_dyn_info = { "size": {}, "urdf_full_path": {} }  # obj dynamic info: size, urdf path, etc.
        self.obj_id_reverse_mapping = {}
        ## obj_id_reverse_mapping: a reverse mapping dict that maps object unique id to:
        # 1. object_name appended with color name
        # 2. object_texture entry in TexturePedia
        # 3. object_description entry in ObjPedia
        self.obj_support_tree = Node(-1)  # -1 refers to the base

        # Configure pybullet
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.dt = 1 / 480
        self.sim_step = 0

        # setup modalities
        modalities = modalities or ["rgb", "segm"]
        if isinstance(modalities, str):
            modalities = [modalities]
        assert set(modalities).issubset(
            {"rgb", "depth", "segm"}
        ), f"Unsupported modalities provided {modalities}"
        # assert "depth" not in modalities, "FIXME: fix depth normalization"
        self.modalities = modalities

        # Setup camera
        self.obs_img_size = obs_img_size
        self.obs_img_views = obs_img_views
        self.set_up_camera(obs_img_size, obs_img_views)
        self.oracle_cams = Oracle.CONFIG

        # Workspace bounds.
        self.pix_size = 0.003125  # 0.003125 m/pixel
        self.bounds = np.array([[0.2, 1.0], [-0.4, 0.4], [0.0, 0.3]])  # Square bounds
        self.zone_bounds = np.copy(self.bounds)
        self.ws_map_size = (
            int(np.round((self.bounds[1, 1] - self.bounds[1, 0]) / self.pix_size)),
            int(np.round((self.bounds[0, 1] - self.bounds[0, 0]) / self.pix_size)),
        )  # workspace map size (height, width)
        self.buffer_shift = np.array([10.0, 10.0, 0.0])  #  A buffer zone for object storage

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
                textArgument=self.assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client,
            )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(self.assets_root, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)
        p.setTimeStep(1.0 / hz, physicsClientId=self.client_id)

        # If display debug window, move default camera closer to the scene.
        if display_debug_window:
            target = p.getDebugVisualizerCamera(physicsClientId=self.client_id)[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,
                physicsClientId=self.client_id,
            )

        assert max_sim_steps_to_static > 0
        self._max_sim_steps_to_static = max_sim_steps_to_static
        self.prompt, self.prompt_assets = None, None
        self.goal_specification = {}  # for StructDiffusion
        self.meta_info = {}
        self._debug = debug
        self._display_debug_window = display_debug_window
        self._hide_arm_rgb = hide_arm_rgb
        self.set_task(task, task_kwargs)
        self.set_seed(seed)

        # setup action space
        self.position_bounds = gym.spaces.Box(
            low=np.array([self.bounds[0, 0], self.bounds[1, 0], self.bounds[2, 0]], dtype=np.float32),
            high=np.array([self.bounds[0, 1], self.bounds[1, 1], self.bounds[2, 1]], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Dict(
            {
                "pose0_position": self.position_bounds,
                "pose0_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
                "pose1_position": self.position_bounds,
                "pose1_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
    
    def connect_pybullet_hook(self, display_debug_window: bool):
        return p.connect(p.DIRECT if not display_debug_window else p.GUI)

    def reset(self):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.obj_dyn_info = { "size": {}, "urdf_full_path": {} }
        self.obj_id_reverse_mapping = {}
        self.obj_support_tree = Node(-1)
        self.meta_info = {}
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
        self.ee = self.task.ee(
            self.assets_root,
            self.ur5,
            9,
            self.obj_ids,
            self.client_id,
        )
        self.ee.is_visible = not self._hide_arm_rgb
        self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5, physicsClientId=self.client_id)
        joints = [
            p.getJointInfo(self.ur5, i, physicsClientId=self.client_id)
            for i in range(n_joints)
        ]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(
                self.ur5, self.joints[i], self.homej[i], physicsClientId=self.client_id
            )

        # Reset end effector.
        self.ee.release()
        
        # reset task
        self.task.reset(self)

        # Re-enable rendering.
        if self._display_debug_window:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )

        # Generate meta info
        self.meta_info["n_objects"] = sum(len(v) for v in self.obj_ids.values())
        self.meta_info["difficulty"] = self.task.difficulty_level or "easy"
        self.meta_info["views"] = list(self.agent_cams.keys())
        self.meta_info["modalities"] = self.modalities
        self.meta_info["seed"] = self._env_seed
        self.meta_info["action_bounds"] = {
            "low": self.position_bounds.low,
            "high": self.position_bounds.high,
        }

        # return observation
        obs, _, _, _, _ = self.step()

        return obs

    def step(self, action=None):
        if action is not None:
            assert self.action_space.contains(
                action
            ), f"got {action} instead, action space {self.action_space}"

            pose0 = (action["pose0_position"], action["pose0_rotation"])
            pose1 = (action["pose1_position"], action["pose1_rotation"])

            if isinstance(self.ee, Suction):
                timeout, released = self.task.primitive(
                    self.movej, self.movep, self.ee, pose0, pose1
                )
            elif isinstance(self.ee, Spatula):
                timeout = self.task.primitive(
                    self.movej, self.movep, self.ee, pose0, pose1
                )
            else:
                raise ValueError("Unknown end effector type")

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self.get_obs()
                return obs, 0.0, True, self._get_info()

        # Step simulator asynchronously until objects settle.
        self.wait_until_settle()

        # check if done
        if isinstance(self.ee, Suction):
            if action is not None:
                result_tuple = self.task.check_success(release_obj=released)
            else:
                result_tuple = self.task.check_success(release_obj=False)
        elif isinstance(self.ee, Spatula):
            result_tuple = self.task.check_success()
        else:
            raise NotImplementedError()
        done = result_tuple.success
        obs = self.get_obs()
        return obs, 0, done, None, self._get_info()
    
    def step_simulation(self):
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_counter += 1
    
    def wait_until_settle(self):
        counter = 0
        while not self.is_static:
            self.step_simulation()
            if counter > self._max_sim_steps_to_static:
                print(
                    f"WARNING: step until static exceeds max {self._max_sim_steps_to_static} steps!"
                )
                break
            counter += 1

    @property
    def task(self) -> BaseTask:
        return self._task

    @property
    def task_name(self) -> str:
        return self.task.task_name

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(p.getBaseVelocity(i, physicsClientId=self.client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def seed(self):
        return self._env_seed

    @property
    def rng(self):
        return self._random

    # Gym functions
    def close(self):
        p.disconnect(physicsClientId=self.client_id)

    def set_seed(self, seed=None):
        self._random = np.random.default_rng(seed=seed)
        self._env_seed = seed
        self.task.set_seed(seed)
        return seed

    # Render & obs
    def set_up_camera(self, obs_img_size: tuple[int, int], obs_img_views: list[str]):
        obs_img_views = obs_img_views or ["front", "top"]
        all_cam_config = get_agent_cam_config(obs_img_size)
        self.agent_cams = {view: all_cam_config[view] for view in obs_img_views}

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

    def get_true_image(self):
        """Get RGB-D orthographic heightmaps and segmentation masks."""

        # Capture near-orthographic RGB-D images and segmentation masks.
        # color: (C, H, W), depth: (1, H, W) within [0, 1]
        color, depth, segm = self.render_camera(self.oracle_cams[0])
        # process images to be compatible with oracle input
        # rgb image, from CHW to HWC
        color = np.transpose(color, (1, 2, 0))
        # depth image, from (1, H, W) to (H, W) within [0, 20]
        depth = 20.0 * np.squeeze(depth, axis=0)

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = misc_utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size
        )

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask

    def _get_info(self):
        result_tuple = self.task.check_success()
        info = {
            "prompt": self.prompt,
            "success": result_tuple.success,
            "failure": result_tuple.failure,
        }
        return info

    def get_obs(self):
        obs = {f"{modality}": {} for modality in self.modalities}  # sensing
        # obs["oracle"] = {}  # oracle information
        obs["point_cloud"] = {}  # point cloud
        obs["poses"] = {}  # object poses
        # Sensing
        for view, config in self.agent_cams.items():
            color, depth, segm = self.render_camera(config)
            render_result = {"rgb": color, "depth": depth, "segm": segm}
            for modality in self.modalities:
                obs[modality][view] = render_result[modality]
            
            # Pointcloud (from top view)
            intrinsic_mat = np.array(config["intrinsics"]).reshape(3, 3)
            # Notice: depth is within [0, 1], so we need to scale it back to [0, 20]
            real_depth = depth[0] * 20.0
            scene_pcd = misc_utils.get_pointcloud(real_depth, intrinsic_mat)
            obs["point_cloud"][view] = {}
            obs["poses"][view] = {}
            #
            max_pcd_size = self.obs_img_size[0] * self.obs_img_size[1]
            obj_pcds = np.zeros([self.max_num_obj * max_pcd_size, 3])
            obj_poses = np.zeros([self.max_num_obj, 7])
            counter = 0
            for obj_id in self.obj_ids["rigid"]:
                # point cloud
                obj_mask = segm == obj_id
                obj_pcd = scene_pcd[obj_mask].reshape(-1, 3)
                # misc_utils.plot_3d(f"{view}-{obj_id}", obj_pcd, color='blue')
                # if obj_pcd.shape[0] == 0:
                #     cv2.imshow("color", color.transpose(1, 2, 0))
                #     cv2.waitKey(0)
                # assert obj_pcd.shape[0] > 0, f"obj_id {obj_id} has no point cloud"
                offset = counter * max_pcd_size
                obj_pcds[offset:offset+obj_pcd.shape[0], :] = obj_pcd

                # object pose
                position, orientation = pybullet_utils.get_obj_pose(self, obj_id)
                offset = counter * 7
                obj_poses[counter, :] = np.array(position + orientation)
                
                counter += 1
            obs["point_cloud"][view] = obj_pcds
            obs["poses"][view] = obj_poses
        # assert self.observation_space.contains(obs)
        return obs

    def get_obj_poses(self):
        """Get the obj poses dict"""
        obj_poses = {}
        for obj_id in self.obj_ids["rigid"]:
            position, orientation = pybullet_utils.get_obj_pose(self, obj_id)
            obj_poses[obj_id] = np.array(position + orientation)
        return obj_poses

    # Add objects related
    @staticmethod
    def _scale_size(size, scalar):
        return (
            tuple([scalar * s for s in size])
            if isinstance(scalar, float)
            else tuple([sc * s for sc, s in zip(scalar, size)])
        )
    
    def get_random_pose(self, obj_size, prior=None, stack_prob=0.0):
        """Get random collision-free object pose within workspace bounds.
        Object has a chance of stack_prob to be stacked upon other objects.
        """

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self.get_true_image()

        obj_stack_id = -1  # -1 is the base
        if self.rng.random() < stack_prob:
            # select an object to stack up
            if len(self.obj_ids["rigid"]) == 0:
                return [None, None], None
            leaf_nodes = self.obj_support_tree.leaves
            leaf_obj_ids = [leaf_node.name for leaf_node in leaf_nodes]
            obj_stack_id = self.rng.choice(leaf_obj_ids)
            # Get the object's top surface
            pos_stack, _ = pybullet_utils.get_obj_pose(self, obj_stack_id)
            obj_stack_size = self.obj_dyn_info["size"][obj_stack_id]
            obj_stack_base = np.array(pos_stack) + np.array([0, 0, obj_stack_size[2] / 2])
            pos = obj_stack_base + np.array([0, 0, obj_size[2] / 2]) + np.array([0, 0, 0.01])  # 1cm above
        else:
            # Randomly sample an object pose within free-space pixels.
            free = np.ones(obj_mask.shape, dtype=np.uint8)
            for obj_ids in self.obj_ids.values():
                for obj_id in obj_ids:
                    free[obj_mask == obj_id] = 0
            free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
            free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
            free = free.astype(np.float32) 
            # Get the probability union
            if prior is not None:
                assert prior.shape == free.shape, "prior shape must be the same as free shape"
                free = np.multiply(free, prior)
            if np.sum(free) == 0:
                return [None, None], None
            pix = misc_utils.sample_distribution(prob=free, rng=self._random)
            pos = misc_utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
            # print(f"pos: {pos}")
            pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = self._random.random() * 2 * np.pi
        rot = misc_utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return [pos, rot], obj_stack_id

    def add_object_to_env(
        self,
        obj_entry: ObjEntry,
        color: TextureEntry,
        size: tuple[float, float, float],
        scalar: float | list[float] = 1.0,
        pose: tuple[tuple, tuple] = None,
        prior: np.ndarray | None = None,  # a prior distribution of object pose
        category: str = "rigid",
        retain_temp: bool = True,
        stack_prob: float = 0.0,
        **kwargs,
    ):
        """helper function for adding object to env."""
        scaled_size = self._scale_size(size, scalar)
        if pose is None:
            pose, obj_stack_id = self.get_random_pose(scaled_size, prior=prior, stack_prob=stack_prob)
        elif pose[0] is None or pose[1] is None:
            # reject sample because of no extra space to use (obj type & size) sampled outside this helper function
            return None, None, None
        else:
            obj_stack_id = None
        obj_id, urdf_full_path = pybullet_utils.add_any_object(
            env=self,
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
        # update support tree
        if obj_stack_id is not None:
            self._update_support_tree(obj_id, obj_stack_id)
        # change texture
        pybullet_utils.p_change_texture(obj_id, color, self.client_id)
        # add mapping info
        pybullet_utils.add_object_id_reverse_mapping_info(
            mapping_dict=self.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=obj_entry,
            texture_entry=color,
        )
        # update dynamic info
        self.obj_dyn_info["size"][obj_id] = scaled_size
        self.obj_dyn_info["urdf_full_path"][obj_id] = urdf_full_path

        return obj_id, urdf_full_path, pose

    def add_random_object_to_env(
        self,
        obj_lists: list[ObjEntry],
        color_lists: list[TextureEntry],
        prior: np.ndarray | None = None,
        stack_prob: float = 0.0,
        **kwargs,
    ):
        """Add random an object from list, with a random texture and random size"""
        sampled_obj = self._random.choice(obj_lists).value
        sampled_obj_size = self._random.uniform(
            low=sampled_obj.size_range.low,
            high=sampled_obj.size_range.high,
        )
        if len(color_lists) > 1:
            sampled_obj_color = self._random.choice(color_lists).value
        elif len(color_lists) == 1:
            sampled_obj_color = color_lists[0].value
        else:
            sampled_obj_color = None
        
        return self.add_object_to_env(
            sampled_obj,
            sampled_obj_color,
            sampled_obj_size,
            prior=prior,
            category="rigid",
            stack_prob=stack_prob,
        )

    # object-level manipulation functions
    def move_all_objects_to_buffer(self):
        """Move all objects to buffer space."""
        for obj_id in self.obj_ids["rigid"]:
            position, orientation = pybullet_utils.get_obj_pose(self, obj_id)
            # apply buffer shift
            position = [_p + _b for _p, _b in zip(position, self.buffer_shift)]
            pybullet_utils.move_obj(self, obj_id, position, orientation)
            # update support tree, objects moved to buffer will be detached from the tree
        self.obj_support_tree.children = []

    def move_object_to_random(self, obj_id: int, prior=None, stack_prob:float=0.0):
        """Move object to a random, free pose inside workspace bounds."""
        obj_size = self.obj_dyn_info["size"][obj_id]
        pose, obj_stack_id = self.get_random_pose(obj_size, prior, stack_prob=stack_prob)
        if pose[0] is None or pose[1] is None:
            return None
        pybullet_utils.move_obj(self, obj_id, pose[0], pose[1])
        self._update_support_tree(obj_id, obj_stack_id)

    def _update_support_tree(self, obj_id, obj_stack_id):
        """Update object support tree."""
        obj_node = None
        obj_stack_node = None
        for __, _, node in RenderTree(self.obj_support_tree):
            if node.name == obj_stack_id:
                obj_stack_node = node
            elif node.name == obj_id:
                obj_node = node
            if obj_stack_node and obj_node:
                obj_node.parent = obj_stack_node
                break
        if obj_stack_node is None:
            raise RuntimeError("obj_stack_node is not found.")
        if obj_node is None:
            Node(obj_id, parent=obj_stack_node)

    # task related
    def set_task(self, task: str | BaseTask, task_kwargs: dict):
        # setup task
        ALL_TASKS = _ALL_TASKS.copy()
        ALL_TASKS.update({k.split("/")[1]: v for k, v in ALL_TASKS.items()})
        if isinstance(task, str):
            assert task in ALL_TASKS, f"Invalid task name provided {task}"
            task = ALL_TASKS[task](debug=self._debug, **(task_kwargs or {}))
        elif isinstance(task, BaseTask):
            task = task
        elif task is None:
            assert False, "No task is set"
        task.assets_root = self.assets_root
        task.set_difficulty("easy")
        self._task = task
        # get agent camera config
        # self.agent_cams = self.task.agent_cams

        # setup action space
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5], dtype=np.float32),
            high=np.array([0.75, 0.50], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Dict(
            {
                "pose0_position": self.position_bounds,
                "pose0_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
                "pose1_position": self.position_bounds,
                "pose1_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
            }
        )

        # Set task-related config
        self.max_num_obj = self.task.max_num_obj

    # ---------------------------------------------------------------------------
    # Debug Functions
    # ---------------------------------------------------------------------------

    def show_support_tree(self):
        print(RenderTree(self.obj_support_tree))

    def save_checkpoint(self, check_point_path: str):
        """Save environment state."""
        save_dir = os.path.dirname(check_point_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        env_state = {}
        env_state["obj_ids"] = self.obj_ids
        env_state["obj_dyn_info"] = self.obj_dyn_info
        env_state["obj_id_reverse_mapping"] = self.obj_id_reverse_mapping
        env_state["obj_support_tree"] = self.obj_support_tree
        env_state["obj_poses"] = self.get_obj_poses()
        env_state["meta_info"] = self.meta_info
        # task related
        env_state["task_state"] = self.task.get_state()
        with open(check_point_path, "wb") as f:
            pickle.dump(env_state, f)

    def load_checkpoint(self, check_point_path: str):
        """Load env from checkpoint."""
        self.reset()  # reset env before loading
        with open(check_point_path, "rb") as f:
            env_state = pickle.load(f)
        # load objects
        obj_ids = env_state["obj_ids"]
        obj_dyn_info = env_state["obj_dyn_info"]
        obj_id_reverse_mapping = env_state["obj_id_reverse_mapping"]
        obj_poses = env_state["obj_poses"]
        for obj_id in obj_ids["rigid"]:
            obj_entry, color_entry = pybullet_utils.recover_obj_and_texture_from_mapping_info(obj_id_reverse_mapping, obj_id)
            obj_size = obj_dyn_info["size"][obj_id]
            obj_pose = (obj_poses[obj_id][:3], obj_poses[obj_id][3:7])  # tuple: (pos, rot)
            self.add_object_to_env(obj_entry, color=color_entry, size=obj_size, pose=obj_pose, category="rigid", retain_temp=False)
            # pybullet_utils.p_change_texture(obj_id, color_entry, self.client_id)
        self.obj_support_tree = env_state["obj_support_tree"]

        # load task
        task_state = env_state["task_state"]
        self.task.set_state(task_state)

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=5):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [
                p.getJointState(self.ur5, i, physicsClientId=self.client_id)[0]
                for i in self.joints
            ]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains,
                physicsClientId=self.client_id,
            )
            self.step_counter += 1
            self.step_simulation()

        print(f"Warning: movej exceeded {timeout} second timeout. Skipping.")
        return True

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints


if __name__ == '__main__':
    assets_root = os.path.join(os.path.dirname(__file__), 'assets')
    scene = BaseEnv(modalities=['rgb'], display_debug_window=True)
    scene.reset()
    
    for i in range(1000):
        scene.step_simulation()
    # Check
    # color = obs['rgb']['top'].transpose(1, 2, 0)
    # cv2.imshow('color', color)
    # cv2.waitKey(0)

    # Test object adding 
    obj_lists = [ObjPedia.BOWL, ObjPedia.BLOCK, ObjPedia.CAPITAL_LETTER_A]
    color_lists = [TexturePedia.RED, TexturePedia.GREEN, TexturePedia.BLUE, TexturePedia.YELLOW]
    for i in range(5):
        scene.add_random_object_to_env(obj_lists, color_lists)
        for i in range(1000):
            scene.step_simulation()

    scene.close()
