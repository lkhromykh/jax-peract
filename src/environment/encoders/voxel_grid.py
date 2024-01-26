import numpy as np
import open3d as o3d
from dm_env import specs

from src.environment import base


class VoxelGrid:

    def __init__(self,
                 scene_bounds: tuple[float, float, float, float, float, float],
                 nbins: int,
                 ) -> None:
        lb, ub = np.split(np.asarray(scene_bounds), 2)
        range_ = ub - lb
        self._scale = lambda x: (x - lb) / range_
        self._shape = lb.size * (nbins,) + (4,)
        self._voxel_size = 1. / (nbins - 1)
        self._bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.zeros_like(lb), np.ones_like(ub))

    def encode(self, obs: base.Observation) -> base.Array:
        def hstack(xs): return np.concatenate([np.reshape(x, (-1, 3)) for x in xs])
        points, colors = map(hstack, (obs['point_clouds'], obs['images']))
        points = self._scale(points)
        colors = colors.astype(np.float32) / 255.
        pcd = o3d.geometry.PointCloud()
        pcd.points, pcd.colors = map(o3d.utility.Vector3dVector, (points, colors))
        pcd = pcd.crop(self._bbox)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, self._voxel_size)
        o3d.visualization.draw_geometries([grid])
        scene = np.zeros(self._shape, dtype=np.uint8)
        for voxel in grid.get_voxels():
            idx = voxel.grid_index
            rgb = np.round(255 * voxel.color)
            scene[tuple(idx)] = np.concatenate([[255], rgb])
        return scene

    def observation_spec(self) -> specs.Array:
        return specs.Array(self._shape, np.uint8, name='voxels')
