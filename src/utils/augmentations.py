"""Diversify training samples via augmentations."""
import tensorflow as tf

import src.types_ as types


def select_random_transition(item: types.Trajectory) -> types.Trajectory:
    act = item['actions']
    tf.debugging.assert_rank(act, 2, message='Leading dim should be time dimension.')
    idx = tf.random.uniform((), minval=0, maxval=tf.shape(act)[0], dtype=tf.int32)
    return tf.nest.map_structure(lambda x: x[idx], item)


def voxel_grid_random_shift(item: types.Trajectory, max_shift: int) -> types.Trajectory:
    """To impose translation invariance."""
    obs = item['observations']
    act = item['actions']
    tf.debugging.assert_rank(act, 1, message='Batching is not supported.')
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
