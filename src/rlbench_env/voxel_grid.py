import numpy as np
import dm_env.specs
import open3d as o3d

from rlbench.backend.observation import Observation

from src import types_ as types

Array = types.Array


class VoxelGrid:

    CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist')

    def __init__(self,
                 scene_bounds: tuple[Array, Array],
                 nbins: int,
                 ) -> None:
        lb, ub = self.scene_bounds = scene_bounds
        range_ = ub - lb
        self._scale = lambda x: (x - lb) / range_
        self.nbins = nbins
        shape = lb.size * (nbins,) + (4,)
        self._scene = np.zeros(shape, dtype=np.uint8)
        self._voxel_size = 1. / (nbins - 1)
        self._bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.zeros_like(lb), np.ones_like(ub))

    def __call__(self, obs: Observation) -> Array:
        def get_view(cam):
            return map(lambda s: getattr(obs, '_'.join([cam, s])),
                       ('point_cloud', 'rgb'))

        points = []
        colors = []
        for cam in self.CAMERAS:
            pcd, rgb = get_view(cam)
            points.append(pcd)
            colors.append(rgb)

        def stack(x): return np.stack(x, 0)
        def reshape(x): return x.reshape((-1, x.shape[-1]))
        def np2o3d(x): return o3d.utility.Vector3dVector(x)
        points, colors = map(
            lambda x: (reshape(stack(x))),
            (points, colors)
        )
        points = self._scale(points)
        points, colors = map(np2o3d, (points, colors))
        pcd = o3d.geometry.PointCloud()
        pcd.points = points
        pcd.colors = colors
        pcd = pcd.crop(self._bbox)
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            self._voxel_size,
            self._bbox.min_bound, self._bbox.max_bound
        )
        self._scene.fill(0)
        for voxel in grid.get_voxels():
            idx = voxel.grid_index
            rgb = voxel.color  # floating precision is lost.
            self._scene[tuple(idx)] = np.concatenate([[255], rgb], -1)
        return self._scene

    def observation_spec(self) -> types.ObservationSpec:
        return dm_env.specs.Array(self._scene.shape, self._scene.dtype,
                                  name='voxels')
