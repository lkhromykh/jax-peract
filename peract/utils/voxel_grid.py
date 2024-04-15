import numpy as np
import open3d as o3d
from dm_env import specs

from peract.environment import gcenv

Array = np.ndarray

try:
    from .voxelize import create_dense_voxel_grid_from_points as cppvoxelize
except ImportError as exc:
    from peract.logger import get_logger
    get_logger().info("Can't import voxelize: %s", exc)

    def create_dense_voxel_grid_from_points(points, colors, scene_bounds, num_voxels):
        lb, ub = scene_bounds
        points = (points - lb) / (ub - lb)
        colors = colors.astype(np.float64) / 255.
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors = map(o3d.utility.Vector3dVector, (points, colors))
        min_bound, max_bound = np.zeros(3), np.ones(3) - 1e-5
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd = pcd.crop(bbox)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                pcd, 1. / num_voxels, min_bound, max_bound)
        scene = np.zeros(3 * (num_voxels,) + (4,), dtype=np.uint8)
        for voxel in grid.get_voxels():
            idx = voxel.grid_index
            rgb = np.round(255 * voxel.color)
            scene[tuple(idx)] = np.r_[rgb, 255]
        return scene
else:
    def create_dense_voxel_grid_from_points(
            points: Array,
            colors: Array,
            scene_bounds: tuple[Array, Array],
            num_voxels: int
    ) -> Array:
        vgrid = cppvoxelize(points, colors, scene_bounds, num_voxels)
        return vgrid.reshape(3 * (num_voxels,) + (4,))


class VoxelGrid:

    def __init__(self,
                 scene_bounds: gcenv.SceneBounds,
                 nbins: int,
                 ) -> None:
        self.scene_bounds = np.split(np.asarray(scene_bounds), 2)
        self.nbins = nbins

    def encode(self, obs: gcenv.Observation) -> Array:
        lb, ub = self.scene_bounds
        points, colors = [], []
        for pcd, rgb in zip(obs.point_clouds, obs.images):
            points.append(pcd.reshape(-1, 3))
            colors.append(rgb.reshape(-1, 3))
        points, colors = map(np.concatenate, (points, colors))
        points = points.astype(np.float64)
        return create_dense_voxel_grid_from_points(points, colors, self.scene_bounds, self.nbins)

    def decode(self, voxels: gcenv.Array) -> o3d.geometry.VoxelGrid:
        grid = o3d.geometry.VoxelGrid()
        grid.voxel_size = 1. / self.nbins
        voxels = voxels.reshape(-1, 4)
        shape = 3 * (self.nbins,)
        for idx, value in enumerate(voxels):
            voxel = o3d.geometry.Voxel()
            rgb, occupied = np.split(value, [3])
            if occupied:
                voxel.grid_index = np.unravel_index(idx, shape)
                voxel.color = rgb.astype(np.float32) / 255.
                grid.add_voxel(voxel)
        return grid

    def observation_spec(self) -> specs.Array:
        return specs.Array(3 * (self.nbins,) + (4,), np.uint8, name='voxels')
