from typing import Literal

import numpy as np
import open3d as o3d
from dm_env import specs

from src.environment import gcenv


# TODO: move loops to C++
class VoxelGrid:

    def __init__(self,
                 scene_bounds: gcenv.SceneBounds,
                 nbins: int,
                 ) -> None:
        self.scene_bounds = lb, ub = np.split(np.asarray(scene_bounds), 2)
        self.shape = lb.size * (nbins,) + (4,)
        self.voxel_size = 1. / nbins
        self._bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.zeros_like(lb), np.ones_like(ub))

    def encode(self,
               obs: gcenv.Observation,
               return_type: Literal['np', 'o3d'] = 'np'
               ) -> gcenv.Array | o3d.geometry.VoxelGrid:
        lb, ub = self.scene_bounds
        points, colors = [], []
        for pcd, rgb in zip(obs.point_clouds, obs.images):
            points.append(pcd.reshape(-1, 3))
            colors.append(rgb.reshape(-1, 3))
        points, colors = map(np.concatenate, (points, colors))
        points = (points - lb) / (ub - lb)
        colors = colors.astype(np.float32) / 255.
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors = map(o3d.utility.Vector3dVector, (points, colors))
        pcd = pcd.crop(self._bbox)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self.voxel_size)
        match return_type:
            case 'o3d':
                return grid
            case 'np':
                scene = np.zeros(self.shape, dtype=np.uint8)
                for voxel in grid.get_voxels():
                    idx = voxel.grid_index
                    if max(idx) < self.shape[0]:
                        rgb = np.round(255 * voxel.color)
                        scene[tuple(idx)] = np.r_[rgb, 255]
                return scene
        raise ValueError(return_type)

    def decode(self, voxels: gcenv.Array) -> o3d.geometry.VoxelGrid:
        grid = o3d.geometry.VoxelGrid()
        grid.voxel_size = self.voxel_size
        voxels = voxels.reshape(-1, 4)
        for idx, value in enumerate(voxels):
            voxel = o3d.geometry.Voxel()
            rgb, occupied = np.split(value, [3])
            if occupied:
                voxel.grid_index = np.unravel_index(idx, self.shape[:-1])
                voxel.color = rgb.astype(np.float32) / 255.
                grid.add_voxel(voxel)
        return grid

    def observation_spec(self) -> specs.Array:
        return specs.Array(self.shape, np.uint8, name='voxels')
