"""Diversify training samples via augmentations."""
from typing import Callable

import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

import src.types_ as types
from src.utils.action_transform import DiscreteActionTransform


def select_random_transition(item: types.Trajectory) -> types.Trajectory:
    act = item.actions
    tf.debugging.assert_rank(act, 2, message='Leading dim should be time dimension.')
    idx = tf.random.uniform((), minval=0, maxval=tf.shape(act)[0], dtype=tf.int32)
    return tf.nest.map_structure(lambda x: x[idx], item)


def scene_rigid_transform_factory(
        action_transform: DiscreteActionTransform,
        max_shift: int
) -> Callable[[types.Trajectory], types.Trajectory]:

    def fn(item: types.Trajectory) -> types.Trajectory:
        item = scene_rotation(item, action_transform)
        item = scene_shift(item, max_shift)
        return item
    return fn


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
                   act_transform: DiscreteActionTransform
                   ) -> types.Trajectory:
    obs, act = _unpack_trajectory(item)
    def cast(x): return tf.cast(x, act.dtype)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    voxels = tf.transpose(obs.voxels, [2, 0, 1, 3])
    new_voxels = tf.image.rot90(voxels, k)
    new_voxels = tf.transpose(new_voxels, [1, 2, 0, 3])
    rot = R.from_rotvec([0, 0, - np.pi * tf.cast(k, tf.float32) / 2])

    act = act_transform.decode(act)
    center = cast(act_transform.get_scene_center())
    pos, tcp_orient, other = tf.split(act, [3, 3, 2])
    rmat = cast(rot.as_matrix())

    new_pos = tf.linalg.matvec(rmat, pos - center) + center
    new_orient = rot.inv() * R.from_euler('ZYX', tcp_orient)
    new_orient = new_orient.as_euler('ZYX')
    new_act = tf.concat([new_pos, new_orient, other], 0)
    new_act = act_transform.encode(new_act)
    return types.Trajectory(
        observations=obs._replace(voxels=new_voxels),
        actions=new_act
    )


def _unpack_trajectory(item: types.Trajectory) -> tuple[types.State, types.Action]:
    obs = item.observations
    act = item.actions
    tf.debugging.assert_rank(act, 1, message='Batching is not supported.')
    return obs, act
