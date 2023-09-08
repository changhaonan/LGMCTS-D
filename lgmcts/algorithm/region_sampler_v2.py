from __future__ import annotations
import lgmcts.utils.misc_utils as utils
"""
New New version of region sampler, remove direction related things.
Instead of using mask, using center + size representation.
Sampling using corrosion now & probability summary now.
FIXME: Currently, there is still a very small error here. Making the result not fully collsion free
"""
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List, Union, Dict
from enum import Enum
import math
import cv2
import copy
import open3d as o3d
import colorsys
import anytree


class SampleStatus(Enum):
    """Sample status"""

    SUCCESS = 0  # success
    REGION_SMALL = 1  # region is too small
    NO_SPACE = 2  # collision
    UNKNOWN = 3  # unknown, placeholder


def sample_distribution(prob, rng, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = prob.flatten() / np.sum(prob)
    # if nnz smaller than n_samples, we sample all of them
    if np.count_nonzero(flat_prob) < n_samples:
        n_samples = np.count_nonzero(flat_prob)
    rand_ind = rng.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False
    )
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    sample_probs = flat_prob[rand_ind]
    return np.int32(rand_ind_coords), sample_probs


def draw_convex_contour(img, pixels):
    """ Draw image, with convex contour """
    bg = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    # convert pixels to cv2 shape
    bg = cv2.drawContours(bg, [pixels], 0, (0, 0, 255), 2)
    return bg


@dataclass
class ObjectData:
    """Object data
    name: name of object
    pos: position in region2D
    pos_offset: in region2D, we use center of 2D mask to represent obj position,
        this can generate a offset compared with real position projection. This
        offset compares the desired origin of object and center of mask in 3d space.
    mask: mask of object, generated by projecting points to 2D plane. Padded to
        have an odd size.
    height: height of object.
    points: points of object.
    color: color of object.
    """
    name: str
    pos: np.ndarray  # center position
    pos_offset: np.ndarray
    mask: np.ndarray  # mask
    height: float  # height
    points: np.ndarray
    color: Tuple[int, int, int]
    rot: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # TODO: currently, rot is not implemented
    collision_mask: np.ndarray = None  # TODO: currently, collision mask is not implemented


@dataclass
class SampleData:
    """Sample representation"""
    pattern: str
    obj_id: int
    obj_ids: list[int]  # all included objects
    obj_poses_pix: dict[int, np.ndarray]  # poses that are already sampled
    sample_info: dict[str, any] = None  # sample information


class Region2DSampler():
    """Region2D sampler"""

    def __init__(
        self,
        resolution: float,
        grid_size: Union[List[int], None] = None,
        world2region: np.ndarray = np.eye(4, dtype=np.float32),
        pix_padding: int = 0,
        pose_boundary: np.ndarray = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
        name: str = "region",
        seed: int = 0,
        **kwargs,
    ):
        """Args:
            pose_boundary: the boundary of pose, used for sampling. Pose can not exceed this boundary.
        """
        self.resolution = resolution
        self.grid_size = grid_size
        self.world2region = world2region  # world to region transform
        self.objects: Dict[int, ObjectData] = {}
        self.obj_support_tree: anytree.Node = None  # support structure
        self.rng = np.random.default_rng(seed=seed)
        self.pix_padding = pix_padding  # padding in pixel
        self.pose_boundary = pose_boundary  # (3, 2) min, max
        self.scene_pcd = None  # scene point cloud

    def reset(self):
        """Reset"""
        self.objects = {}
        self.obj_support_tree = None

    def _pix2world(self, pos: np.ndarray):
        """Transform position from pix to world"""
        if pos.shape == (3,):
            pos = pos[None, :]
        # use homogeneous coordinate
        if pos.shape[-1] != 3:
            pos = pos[:, :3]
        # there will be a small error in the position
        pos = (pos[:, :3] * self.resolution).astype(np.float32)
        pos = np.hstack((pos, np.ones((pos.shape[0], 1))))
        pos = (np.matmul(np.linalg.inv(self.world2region), pos.T)).T
        return pos[:, :3].reshape(-1, 3)

    def _world2pix(self, pos: np.ndarray):
        """Transform position from world to pix"""
        # append dim
        if pos.shape == (3,):
            pos = pos[None, :]
        # use homogeneous coordinate
        if pos.shape[-1] == 3:
            # pos is of shape (N, 3,)
            pos = np.hstack((pos, np.ones((pos.shape[0], 1))))
        elif pos.shape[-1] != 4:
            raise ValueError("pos'shape should be 3 or 4")
        pos = (self.world2region @ pos.T).T
        pos = (pos[:, :3] / self.resolution).astype(np.int32)
        return pos.squeeze()

    def add_object(
        self,
        obj_id: int,
        points: np.ndarray,
        pos_ref: Union[None, np.ndarray] = None,
        name: str | None = None,
        color=(127, 127, 127),
        mask_mode: str = "sphere"
    ):
        """Add object to scene, create mask from points
        Args:
            mask_mode: "sphere", "raw_mask", "convex_hull". "sphere" is to provide clearance.
        """
        assert points is not None, "points should not be None"
        if pos_ref is None:
            pos_ref = (points.max(axis=0) + points.min(axis=0)) / 2.0
        # project points to region plane
        points_pix = self._world2pix(points)
        lb_pix = np.array(
            [points_pix[:, 0].min(), points_pix[:, 1].min(), points_pix[:, 2].min()]
        )  # lb, lower bottom
        if mask_mode == "sphere":
            mask_height = points_pix[:, 0].max() - points_pix[:, 0].min() + 1
            mask_width = points_pix[:, 1].max() - points_pix[:, 1].min() + 1
            mask_size = math.ceil(math.sqrt(mask_height ** 2 + mask_width ** 2))
            # pad size to odd
            if mask_size % 2 == 0:
                mask_size += 1
            mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
            # draw a filled circle
            cv2.circle(mask, (mask_size // 2, mask_size // 2), mask_size // 2, 1, thickness=-1)
        elif mask_mode == "convex_hull":
            mask_height = points_pix[:, 0].max() - points_pix[:, 0].min() + 1
            mask_width = points_pix[:, 1].max() - points_pix[:, 1].min() + 1
            # pad size to odd
            if mask_width % 2 == 0:
                mask_width += 1
            if mask_height % 2 == 0:
                mask_height += 1
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            points_convex_hull = ConvexHull(points_pix[:, :2])
            pixels = (points_convex_hull.points[points_convex_hull.vertices]).astype(np.int32) - lb_pix[:2]
            cv2.fillConvexPoly(mask, pixels, 1,)
            # DEBUG start here
            # contour = draw_convex_contour(mask, pixels)
            # cv2.imshow("contour", mask * 255)
            # cv2.waitKey(0)
            # DEBUG end here
        elif mask_mode == "raw_mask":
            mask_height = points_pix[:, 0].max() - points_pix[:, 0].min() + 1
            mask_width = points_pix[:, 1].max() - points_pix[:, 1].min() + 1
            # pad size to odd
            if mask_width % 2 == 0:
                mask_width += 1
            if mask_height % 2 == 0:
                mask_height += 1
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            pixels = points_pix[:, :2].astype(np.int32) - lb_pix[:2]
            mask[pixels[:, 0], pixels[:, 1]] = 1
        # ##DEBUG: check mask
        # # resize the height to 500
        # mask_vis = cv2.resize(mask, (mask.shape[1] * 500 // mask.shape[0], 500), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("mask", mask_vis * 255)
        # cv2.waitKey(0)
        height = points_pix[:, 2].max() - points_pix[:, 2].min()
        # compute offset compared with pos_ref (reference position)
        mask_center = np.array([mask.shape[0] // 2, mask.shape[1] // 2, 0]) + lb_pix
        mask_center_world = self._pix2world(mask_center)
        pos_offset = mask_center_world - pos_ref
        name = name if name is not None else f"obj_{obj_id}"
        # apply a safety padding to mask
        mask = cv2.copyMakeBorder(mask, self.pix_padding, self.pix_padding, self.pix_padding,
                                  self.pix_padding, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.dilate(mask, np.ones((self.pix_padding, self.pix_padding), dtype=np.uint8), iterations=1)
        # cv2.imshow("mask", mask * 255)
        # cv2.waitKey(0)
        self.objects[obj_id] = ObjectData(name=name, pos=mask_center, mask=mask,
                                          height=height, color=color, points=points, pos_offset=pos_offset)

    def get_object_pose(self, obj_id: int) -> np.ndarray:
        """Get object position"""
        if obj_id in self.objects:
            mask_center_world = self._pix2world(self.objects[obj_id].pos)
            pos_ref_world = mask_center_world - self.objects[obj_id].pos_offset
            # clip by boundary
            # pos_ref_world = np.clip(pos_ref_world, a_max=self.pose_boundary[:, 1], a_min=self.pose_boundary[:, 0])
            return np.hstack([pos_ref_world[0], self.objects[obj_id].rot])
        else:
            raise ValueError("Object not found")

    def get_object_poses(self) -> Dict[str, np.ndarray]:
        """Get object dict"""
        obj_poses = {}
        for obj_id in self.objects:
            obj_poses[obj_id] = self.get_object_pose(obj_id)
        return obj_poses

    def set_object_pose(
        self, obj_id: int, obj_pos: np.ndarray, enable_vis: bool = False
    ) -> None:
        """Update object in scene"""
        assert obj_id in self.objects, "Object not found"
        mask_center_world = obj_pos[:3] + self.objects[obj_id].pos_offset  # position
        # mask_center_world = obj_pos[:3]
        if obj_id in self.objects:
            self.objects[obj_id].pos = self._world2pix(mask_center_world)
            if enable_vis:
                self.visualize()

    def set_object_poses(
        self, obj_states: Dict[int, np.ndarray], enable_vis: bool = False
    ) -> None:
        """Update all fg objects in scene"""
        for obj_id, obj_pose in obj_states.items():
            self.set_object_pose(obj_id, obj_pose, enable_vis)

    def _put_mask(
        self,
        mask: np.ndarray,
        pos: np.ndarray,
        occupancy_map: np.ndarray,
        **kwargs,
    ) -> bool:
        """Put mask to the occupancy grid, pos is at left bottom corner of mask"""
        height, width = occupancy_map.shape[:2]
        mask_x = mask.shape[0]
        mask_y = mask.shape[1]
        mask_half_x = (mask_x - 1) // 2
        mask_half_y = (mask_y - 1) // 2

        # put mask
        value = kwargs.get("value", 1.0)
        # compute the mask remaining in the region
        mask_min_x = max(0, mask_half_x - pos[0])
        mask_max_x = min(mask_x, height - pos[0] + mask_half_x)
        mask_min_y = max(0, mask_half_y - pos[1])
        mask_max_y = min(mask_y, width - pos[1] + mask_half_y)
        if mask_max_x <= mask_min_x or mask_max_y <= mask_min_y:
            return False  # no mask in region
        mask_in_region = mask[mask_min_x:mask_max_x, mask_min_y:mask_max_y]
        assert len(occupancy_map.shape) == 3, "Only support 3D occupancy map"
        occupancy_map[
            pos[0] - mask_half_x + mask_min_x: pos[0] - mask_half_x + mask_max_x,
            pos[1] - mask_half_y + mask_min_y: pos[1] - mask_half_y + mask_max_y,
            :,
        ][mask_in_region == 1] = value
        # DEBUG
        cv2.circle(occupancy_map, (pos[1], pos[0]), 1, (255, 0, 0), thickness=-1)
        cv2.rectangle(occupancy_map, (pos[1] - mask_half_y, pos[0] - mask_half_x),
                      (pos[1] + mask_half_y, pos[0] + mask_half_x), (0, 255, 0), thickness=1)
        return True

    def get_occupancy(self, obj_list: list[int] | None = None) -> bool:
        """Update occupancy grid occupied by obj_list"""
        occupancy_map = np.ones((self.grid_size[0], self.grid_size[1], 1), dtype=np.float32)
        # objects
        if obj_list is None:
            obj_list = list(self.objects.keys())
        for obj_id, obj_data in self.objects.items():
            if obj_id in obj_list:
                self._put_mask(
                    mask=obj_data.mask,
                    pos=obj_data.pos,
                    occupancy_map=occupancy_map,
                    value=0.0,
                )
        return occupancy_map

    def get_free_space(self, obj_id: int, allow_outside: bool = True) -> np.ndarray:
        """Get the free space of the object using cv2.erosion"""
        obj = self.objects[obj_id]
        mask = obj.mask
        obj_list = [id for id in self.objects if id != obj_id]
        occupancy_map = self.get_occupancy(obj_list)
        if not allow_outside:
            # mark the boundary as 0
            occupancy_map[0, :, :] = 0
            occupancy_map[-1, :, :] = 0
            occupancy_map[:, 0, :] = 0
            occupancy_map[:, -1, :] = 0
        # get free space, free is 1, occupied is 0
        kernel_size = max(mask.shape[0], mask.shape[1])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        free_space = cv2.erode(occupancy_map, kernel, iterations=1)

        # ## DEBUG
        # scene_image = self.project_pcd(self.scene_pcd)
        # # augment free_space to 3 channels
        # free_space_3c = (np.concatenate([free_space[:, :, None], free_space[:, :, None],
        #                  free_space[:, :, None]], axis=2) * 255).astype(np.uint8)
        # assert scene_image.shape == free_space_3c.shape, "scene image and occupancy grid should have the same size"
        # alpha = 0.5  # Adjust this value as needed (0.0 for fully transparent, 1.0 for fully opaque)
        # overlay = cv2.addWeighted(scene_image, 1 - alpha, free_space_3c, alpha, 0)
        # # resize to height 300
        # overlay = cv2.resize(overlay, (overlay.shape[1] * 300 //
        #                      overlay.shape[0], 300), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("free_space", overlay * 255)
        # cv2.waitKey(0)
        return free_space

    def sample(
        self, obj_id: int, n_samples: int, prior: np.array | None = None, allow_outside: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], SampleStatus, Dict[str, any]]:
        """General sampling method
        Args:
            allow_outside: whether allow sampling outside the region (paritially)
        Return:
            samples_3d: samples in 3D
            samples_2d: samples in 2D (pixel)
            sample_status: sample status
            sample_info: information of sampling
                - free_volume: free volume of the region
                - sample_probs: probability of each sample
        """
        free_space = self.get_free_space(obj_id, allow_outside).astype(np.float32)  # free is 1, occupied is 0
        if prior is not None:
            assert prior.shape[:2] == free_space.shape[:2], "prior shape must be the same as free shape"
            free_space = np.multiply(free_space, prior)
        if np.sum(free_space) == 0:
            return np.array([]), np.array([]), SampleStatus.NO_SPACE, {}
        samples_pix, sample_probs = sample_distribution(prob=free_space, rng=self.rng, n_samples=n_samples)  # (N, 2)
        samples_pix = np.concatenate([samples_pix, np.zeros((samples_pix.shape[0], 1))], axis=1)  # (N, 3)
        samples_wd = self._pix2world(samples_pix)  # (N, 3)
        samples_wd = samples_wd - self.objects[obj_id].pos_offset.reshape(1, 3)
        # samples_wd = np.clip(samples_wd, a_max=self.pose_boundary[:, 1], a_min=self.pose_boundary[:, 0])

        # FIXME: currently we don't support sample in rotation, so we set it to identity
        rots = np.tile(np.array([0.0, 0.0, 0.0, 1.0-(1e-6)], dtype=np.float32), (samples_pix.shape[0], 1))
        samples_wd = np.hstack([samples_wd, rots])
        # Assemble sample info
        sample_info = {
            "free_volume": np.sum(free_space),
            "sample_probs": sample_probs,
        }
        return samples_wd, samples_pix, SampleStatus.SUCCESS, sample_info

    # Debug related methods
    def project_pcd(self, pcd):
        """Convert point cloud to image"""
        # convert to image
        pcd_np = np.array(pcd.points)
        pcd_np = self._world2pix(pcd_np)
        pcd_np = np.clip(pcd_np, a_max=np.array([self.grid_size[0] - 1, self.grid_size[1] - 1, self.grid_size[2] - 1]),
                         a_min=np.zeros(3,)).astype(np.int32)
        color = np.array(pcd.colors) * 255
        color[:, [0, 1, 2]] = color[:, [2, 1, 0]]  # convert to BGR
        # convert to image
        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        img[pcd_np[:, 0], pcd_np[:, 1], :] = color
        return img

    def visualize(self, **kwargs):
        """Visualize the occupancy grid"""
        # visualize
        img = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)

        # draw the objects
        for obj_id, obj_data in self.objects.items():
            obj_color_np = np.array(obj_data.color)
            try:
                self._put_mask(
                    mask=obj_data.mask,
                    pos=obj_data.pos,
                    offset=None,
                    occupancy_map=img,
                    value=obj_color_np,
                )
            except:
                continue
        # concat with scene image
        if self.scene_pcd is not None:
            scene_image = self.project_pcd(self.scene_pcd)
            assert scene_image.shape == img.shape, "scene image and occupancy grid should have the same size"
            alpha = 0.5  # Adjust this value as needed (0.0 for fully transparent, 1.0 for fully opaque)
            overlay = cv2.addWeighted(scene_image, 1 - alpha, img, alpha, 0)
            img = np.concatenate([img, scene_image, overlay], axis=1)

        # resize the image
        vis_img_size = 300
        img_resized = cv2.resize(img, (img.shape[1] * vis_img_size // img.shape[0], vis_img_size),
                                 interpolation=cv2.INTER_NEAREST)
        show_grid = kwargs.get("show_grid", False)
        if show_grid:
            grid_color = (255, 255, 255, 125)  # White color for grid, 125 for transparency
            for i in range(0, img_resized.shape[0], scale_factor):
                cv2.line(img_resized, (0, i), (img_resized.shape[1], i), grid_color, 1)
            for j in range(0, img_resized.shape[1], scale_factor):
                cv2.line(img_resized, (j, 0), (j, img_resized.shape[0]), grid_color, 1)

        # cv2.imshow("occpuancy-scene-overlay", img_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def visualize_3d(self, show_origin: bool = False):
        """Visualize the region and obj bbox in 3D"""
        vis_list = []
        if self.scene_pcd is not None:
            vis_list.append(self.scene_pcd)
        # get obj bbox
        for obj_id, obj_data in self.objects.items():
            o3d_color = (obj_data.color[0] / 255.0, obj_data.color[1] / 255.0, obj_data.color[2] / 255.0)
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(obj_data.points)
            )
            bbox.color = o3d_color
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_data.points))
            pcd.paint_uniform_color(o3d_color)
            # transform obj to global pos
            obj_pose = self.get_object_pose(obj_id)
            pcd.translate(obj_pose[:3])
            bbox.translate(obj_pose[:3])
            vis_list.append(bbox)
            vis_list.append(pcd)
        if show_origin:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis_list.append(origin)
            region2world = np.linalg.inv(self.world2region)
            region_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            region_origin.transform(region2world)
            vis_list.append(region_origin)
        o3d.visualization.draw_geometries(vis_list)


################ LGMCTS related ################


class Region2DSamplerLGMCTS(Region2DSampler):
    """Region sampler wrapper for LGMCTS"""

    def __init__(self, resolution: float, pix_padding: int, bounds: np.ndarray):
        """Here we only the world-coordinate is the camera-coordinate. The region coordinate starts at
        the left-top corner of the region, with x-axis pointing down, y-axis pointing right
        """
        grid_size = (int((bounds[0, 1] - bounds[0, 0]) / resolution),
                     int((bounds[1, 1] - bounds[1, 0]) / resolution),
                     int((bounds[2, 1] - bounds[2, 0]) / resolution))

        region2world = np.eye(4, dtype=np.float32)
        region2world[:, 0] = np.array([0, 1, 0, 0])
        region2world[:, 1] = np.array([1, 0, 0, 0])
        region2world[:, 2] = np.array([0, 0, -1, 0])
        region2world[:, 3] = np.array([bounds[1, 0], bounds[0, 0], -bounds[2, 0], 1])
        # pose boundary
        pose_boundary = bounds
        super().__init__(resolution, grid_size, world2region=np.linalg.inv(
            region2world), pix_padding=pix_padding, pose_boundary=pose_boundary)

    def load_env(self, env, mask_mode: str, **kwargs):
        """Load objects from observation"""
        # load objects
        obs = env.get_obs()
        obj_pcds = obs["point_cloud"]["top"]
        obj_poses = obs["poses"]["top"]
        obj_lists = env.obj_ids["rigid"]
        obj_names = [env.obj_id_reverse_mapping[obj_id]["obj_name"] for obj_id in obj_lists]
        max_pcd_size = env.obs_img_size[0] * env.obs_img_size[1]
        obj_lists, obj_pcd_list, obj_pose_list = utils.separate_pcd_pose(obj_lists, obj_pcds, obj_poses, max_pcd_size)
        # get color
        color = obs["rgb"]["top"].transpose((1, 2, 0))
        segm = obs["segm"]["top"]

        name_ids = [(name, id) for name, id in zip(obj_names, obj_lists)]
        self.load_from_pcds(obj_pcd_list, name_ids, mask_mode, **kwargs)
        self.obj_support_tree = env.obj_support_tree

    def load_from_pcds(self, pcd_list: list, name_ids: list, mask_mode: str, **kwargs):
        """Load from pcd (in open3d representation) list"""
        # get scene_pcd
        self.scene_pcd = o3d.geometry.PointCloud()
        pcd_np_list = []
        color_list = []
        if isinstance(pcd_list[0], np.ndarray):
            if "color" in kwargs:
                color_list = kwargs["color"]
            else:
                color_list = [np.random.rand(3) for _ in range(len(pcd_list))]
            for i, pcd in enumerate(pcd_list):
                obj_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
                obj_pcd.paint_uniform_color(color_list[i])
                self.scene_pcd += obj_pcd
            pcd_np_list = pcd_list
        elif isinstance(pcd_list[0], o3d.geometry.PointCloud):
            for pcd in pcd_list:
                self.scene_pcd += pcd
            pcd_np_list = [np.asarray(pcd.points) for pcd in pcd_list]
            color_list = [np.asarray(pcd.colors) for pcd in pcd_list]

        for i, (obj_pcd, obj_color, name_id) in enumerate(zip(pcd_np_list, color_list, name_ids)):
            name, id = name_id
            obj_pose_max = obj_pcd.max(axis=0)
            obj_pose_min = obj_pcd.min(axis=0)
            obj_pcd_center = (obj_pose_max + obj_pose_min) / 2.0
            obj_pcd -= obj_pcd_center
            pos_ref = None
            obj_color = obj_color.mean(axis=0) * 255.0
            # add object to region sampler
            self.add_object(obj_id=id, points=obj_pcd, pos_ref=pos_ref, name=name, color=obj_color, mask_mode=mask_mode)
            self.set_object_pose(obj_id=id, obj_pos=obj_pcd_center)
            now_pose = self.get_object_pose(id)
        self.obj_support_tree = None
