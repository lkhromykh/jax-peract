import numpy as np
import open3d as o3d
from dm_env import specs

from src.environment import gcenv


class VoxelGrid:

    def __init__(self,
                 scene_bounds: gcenv.SceneBounds,
                 nbins: int,
                 ) -> None:
        lb, ub = np.split(np.asarray(scene_bounds), 2)
        range_ = ub - lb
        self._scale = lambda x: (x - lb) / range_
        self._shape = lb.size * (nbins,) + (4,)
        self._voxel_size = 1. / (nbins - 1)
        self._bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.zeros_like(lb), np.ones_like(ub))

    def encode(self,
               obs: gcenv.Observation,
               return_o3d_geoms: bool = False  # for visualization
               ) -> gcenv.Array | tuple[o3d.geometry.PointCloud, o3d.geometry.VoxelGrid]:
        points = self._scale(obs.point_clouds).reshape(-1, 3)
        colors = obs.images.reshape(-1, 3).astype(np.float32) / 255.
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors = map(o3d.utility.Vector3dVector, (points, colors))
        pcd = pcd.crop(self._bbox)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self._voxel_size)
        if return_o3d_geoms:
            return pcd, grid
        scene = np.zeros(self._shape, dtype=np.uint8)
        for voxel in grid.get_voxels():
            idx = voxel.grid_index
            rgb = np.round(255 * voxel.color)
            scene[tuple(idx)] = np.r_[rgb, 255]
        return scene

    def decode(self, voxels: gcenv.Array) -> o3d.geometry.VoxelGrid:
        grid = o3d.geometry.VoxelGrid()
        grid.voxel_size = self._voxel_size
        voxels = voxels.reshape(-1, 4)
        for idx, value in enumerate(voxels):
            voxel = o3d.geometry.Voxel()
            rgb, occupied = np.split(value, [3])
            if occupied:
                voxel.grid_index = np.unravel_index(idx, self._shape[:-1])
                voxel.color = rgb.astype(np.float32) / 255.
                grid.add_voxel(voxel)
        return grid

    def observation_spec(self) -> specs.Array:
        return specs.Array(self._shape, np.uint8, name='voxels')
