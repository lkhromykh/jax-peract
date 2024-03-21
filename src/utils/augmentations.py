"""Diversify training samples via augmentations."""
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

import src.types_ as types
from src.utils.action_transform import DiscreteActionTransform


def scene_shift(item: types.Trajectory, max_shift: int) -> types.Trajectory:
    """To impose translation invariance."""
    obs, act = _unpack_trajectory(item)
    vgrid = obs.voxels
    *size, channels = vgrid.shape
    pos, low_dim = tf.split(act, [l := len(size), tf.size(act) - l])
    size = tf.convert_to_tensor(size)
    shift = tf.random.uniform(size.shape, -max_shift, max_shift + 1, tf.int32)
    shift = tf.clip_by_value(shift, -pos, size - 1 - pos)
    def append(x, vals): return tf.concat([x, tf.constant(vals, x.dtype)], 0)
    padding = tf.fill((tf.size(shift), 2), max_shift)
    vgrid = tf.pad(vgrid, append(padding, [[0, 0]]))
    vgrid = tf.slice(
        vgrid,
        begin=append(max_shift - shift, [0]),
        size=append(size, [channels])
    )
    pos += shift
    return types.Trajectory(observations=obs._replace(voxels=vgrid),
                            actions=tf.concat([pos, low_dim], 0))


def scene_rotation(item: types.Trajectory,
                   act_transform: DiscreteActionTransform,
                   ) -> types.Trajectory:
    observation, action = _unpack_trajectory(item)
    vgrid_shape, act_shape = observation.voxels.shape, action.shape

    def np_rot(voxels, act):
        k = np.random.randint(0, 4)
        if k == 0:
            return voxels, act
        new_voxels = np.rot90(voxels, k, axes=(0, 1))
        rot = R.from_rotvec([0, 0, np.pi * k / 2])
        act = act_transform.decode(act)
        center = act_transform.get_scene_center()
        pos, tcp_orient, other = np.split(act, [3, 6])
        rmat = rot.as_matrix()
        new_pos = rmat @ (pos - center) + center
        new_orient = rot * R.from_euler('ZYX', tcp_orient)
        new_orient = new_orient.as_euler('ZYX')
        new_act = np.concatenate([new_pos, new_orient, other])
        new_act = act_transform.encode(new_act)
        return new_voxels, new_act

    voxels_, act_ = tf.numpy_function(
        func=np_rot,
        inp=[observation.voxels, action],
        Tout=[tf.uint8, tf.int32],
        stateful=False
    )
    voxels_ = tf.ensure_shape(voxels_, vgrid_shape)
    act_ = tf.ensure_shape(act_, act_shape)
    traj = types.Trajectory(observations=observation._replace(voxels=voxels_), actions=act_)
    return tf.nest.map_structure(tf.convert_to_tensor, traj)


def color_transforms(item: types.Trajectory,
                     max_brightness: float = 0.1,
                     contrast: float = 0.1,
                     saturation: float = 0.1,
                     hue: float = 0.02
                     ) -> types.Trajectory:
    obs, act = item.observations, item.actions
    colors, occupancy = tf.split(obs.voxels, [3, 1], -1)
    if max_brightness > 0:
        colors = tf.image.random_brightness(colors, max_brightness)
    if contrast > 0:
        colors = tf.image.random_contrast(colors, 1 - contrast, 1 + contrast)
    if saturation > 0:
        colors = tf.image.random_saturation(colors, 1 - saturation, 1 + saturation)
    if hue > 0:
        colors = tf.image.random_hue(colors, hue)
    colors *= tf.cast(occupancy == 255, tf.uint8)

    obs = obs._replace(voxels=tf.concat([colors, occupancy], -1))
    return types.Trajectory(observations=obs, actions=act)


def _unpack_trajectory(item: types.Trajectory) -> tuple[types.State, types.Action]:
    obs = item.observations
    act = item.actions
    tf.debugging.assert_rank(act, 1, message='Batching is not supported.')
    return obs, act
