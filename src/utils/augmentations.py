"""Diversify training samples via augmentations."""
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

import src.types_ as types
from src.utils.action_transform import DiscreteActionTransform


def select_random_transition(item: types.Trajectory) -> types.Trajectory:
    act = item['actions']
    tf.debugging.assert_rank(act, 2, message='Leading dim should be time dimension.')
    idx = tf.random.uniform((), minval=0, maxval=tf.shape(act)[0], dtype=tf.int32)
    return tf.nest.map_structure(lambda x: x[idx], item)


def se3_random_transform_factory(action_transform: DiscreteActionTransform, max_shift: int):
    def fn(item: types.Trajectory) -> types.Trajectory:
        item = scene_rotation(item, action_transform)
        item = scene_mirroring(item, action_transform)
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


# TODO: mirroring, rotations augmentations
def scene_rotation(item: types.Trajectory,
                   act_transform: DiscreteActionTransform
                   ) -> types.Trajectory:
    obs, act = _unpack_trajectory(item)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    rot = R.from_rotvec([0, 0, - np.pi * tf.cast(k, tf.float32) / 2])
    new_voxels = tf.experimental.numpy.rot90(obs.voxels, k, axes=(0, 1))
    new_act = act_transform.decode(act)
    new_act = _rotate_action(rot, new_act)
    new_act = act_transform.encode(new_act)
    return types.Trajectory(
        observations=obs._replace(voxels=new_voxels),
        actions=new_act
    )


def scene_mirroring(item: types.Trajectory,
                    act_transform: DiscreteActionTransform
                    ) -> types.Trajectory:
    obs, act = _unpack_trajectory(item)
    do_inverse = tf.random.uniform(()) > 0.5
    new_voxels = tf.where(do_inverse, tf.reverse(obs.voxels, [0]), obs.voxels)
    rot = tf.linalg.diag([1 - 2 * tf.cast(do_inverse, tf.float32), 1, 1])
    rot = R.from_matrix(rot)
    new_act = act_transform.decode(act)
    new_act = _rotate_action(rot, new_act)
    new_act = act_transform.encode(new_act)
    return types.Trajectory(
        observations=obs._replace(voxels=new_voxels),
        actions=new_act
    )


def _rotate_action(rot: R, action: tf.Tensor) -> tf.Tensor:
    dtype = action.dtype
    center = 0.5 * tf.ones(3, dtype=dtype)
    pos, gripper_rot, other = tf.split(action, [3, 3, 2])
    rmat = tf.cast(rot.as_matrix(), dtype)
    new_pos = tf.linalg.matvec(rmat, pos - center) + center
    new_grot = rot.inv() * R.from_euler('ZYX', gripper_rot)
    return tf.cast(tf.concat([new_pos, new_grot.as_euler('ZYX'), other], 0), dtype)


def _unpack_trajectory(item: types.Trajectory) -> tuple[types.State, types.Action]:
    obs = item['observations']
    act = item['actions']
    tf.debugging.assert_rank(act, 1, message='Batching is not supported.')
    return obs, act
