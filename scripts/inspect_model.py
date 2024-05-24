import os
import sys
import pathlib
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import jax
import jax.numpy as jnp

from peract.config import Config
from peract.builder import Builder


# TODO: vizualize attention, next action
def inspect_model(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    vgrid = enc.scene_encoder
    nets, _ = builder.make_networks_and_params(enc)
    params = builder.load(Builder.STATE).params
    ds = builder.make_tfdataset('val')
    ds = ds.unbatch().shuffle(1000).as_numpy_iterator()

    def propose_actions(obs_):
        dist = nets.apply(params, obs_)
        modal_action = dist.mode()
        pos_dist = dist.distributions[0]
        logits, idxs = jax.lax.top_k(pos_dist.distribution.logits, 7)
        probs = jax.nn.softmax(logits)
        actions = map(pos_dist.bijector.forward, idxs)
        return modal_action, (jnp.stack(list(actions)), probs)

    def _viz_action(action_, is_expert: bool = False):
        _, grip, term = np.split(action_, [6, 7])
        color = np.array([0, 0, 1.]) if grip < 0.5 else np.array([0., 1., 0.])
        if term > 0.5:
            color = np.array([1., 0, 0])
        voxel = o3d.geometry.Voxel(
            grid_index=action_[:3],
            color=color
        )
        action_ = enc.action_encoder.decode(action_)
        tcp_pos, tcp_rot, _ = np.split(action_, [3, 6])
        lb, ub = vgrid.scene_bounds
        tcp_pos = (tcp_pos - lb) / (ub - lb)
        size = 0.2 if is_expert else 0.1
        frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=tcp_pos)
        frame_tcp = frame_tcp.rotate(R.from_euler('ZYX', tcp_rot).as_matrix())
        return voxel, frame_tcp

    def viz_one(sample):
        obs = sample.observations
        expert_action = sample.actions
        out = jax.jit(propose_actions)(obs)
        modal_action, (top_k_actions, probs) = jax.device_get(out)
        modal_voxel, modal_tcp = _viz_action(modal_action)
        exp_voxel, exp_tcp = _viz_action(expert_action, is_expert=True)
        voxels = vgrid.decode(obs.voxels)
        for act, prob in zip(top_k_actions, probs):
            color = (1 + 2 * prob) / 3 * modal_voxel.color
            voxel = o3d.geometry.Voxel(
                grid_index=act,
                color=color
            )
            voxels.add_voxel(voxel)
        voxels.add_voxel(modal_voxel), voxels.add_voxel(exp_voxel)
        o3d.visualization.draw_geometries([voxels, modal_tcp, exp_tcp])

    try:
        for sample_ in ds:
            viz_one(sample_)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    path = pathlib.Path(sys.argv[1]).resolve()
    cfg = Config.load(path / Builder.CONFIG, compute_dtype='f32')
    print('Reminder: blue is move, green is grasp, red is termination.')
    inspect_model(cfg)
