import os
import sys
import random
import pathlib
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
plt.rcParams['animation.embed_limit'] = 2 ** 20

from src.environment import gcenv
from src.utils.voxel_grid import VoxelGrid
from src.dataset.dataset import DemosDataset
from src.dataset.keyframes_extraction import extractor_factory


def viz_demo(demo: gcenv.Demo) -> animation.FuncAnimation:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), squeeze=True, tight_layout=True)
    obs_ax, kf_ax, jvel_ax, gpos_ax = axes.flatten()
    pairs, kf_idxs = extractor_factory()(demo)

    next_kf_idxs = []
    kfs = iter(kf_idxs)
    cur_kf = next(kfs)
    for t in range(kf_idxs[-1]):
        if t == cur_kf:
            cur_kf = next(kfs)
        next_kf_idxs.append(cur_kf)
    next_kf_idxs.append(cur_kf)

    obs_ax.set_title('Observation')
    kf_ax.set_title('Next keyframe')
    jvel_ax.set_title('Joints max velocity')
    gpos_ax.set_title('Gripper pos')
    obs_ax.axis('off')
    kf_ax.axis('off')

    def as_img(obs): return obs.images[0]

    obs0, kf0 = pairs[0]
    fig.suptitle(f"{obs0.goal}\nDemo length: {len(pairs)}, keyframes_indices: {kf_idxs}")
    obs_art = obs_ax.imshow(as_img(obs0))
    kf_art = kf_ax.imshow(as_img(kf0))
    jvel_ax.plot([max(abs(obs.joint_velocities)) for obs in demo])
    jvel_ax.axhline(obs0.JOINTS_VEL_LOW_THRESHOLD, c='r', label='low_velocity_threshold')
    jvel_obs_line = jvel_ax.axvline(0, c='k')
    jvel_kf_line = jvel_ax.axvline(kf_idxs[0], c='g')
    jvel_ax.set_xticks(kf_idxs)
    jvel_ax.set_xlim(left=0, right=kf_idxs[-1])
    jvel_ax.legend(loc='upper left')

    gpos_ax.plot([obs.gripper_pos for obs in demo])
    gpos_ax.axhline(obs0.GRIPPER_OPEN_THRESHOLD, c='r', label='open_threshold')
    gpos_obs_line = gpos_ax.axvline(0, c='k')
    gpos_kf_line = gpos_ax.axvline(kf_idxs[0], c='g')
    gpos_ax.set_xticks(kf_idxs)
    gpos_ax.set_xlim(left=0, right=kf_idxs[-1])
    gpos_ax.set_ylim(ymin=0, ymax=1.)
    gpos_ax.legend(loc='upper left')

    def viz_slice(idx):
        obs, kf = pairs[idx]
        obs_art.set_data(as_img(obs))
        kf_art.set_data(as_img(kf))
        idx2 = [idx, idx]
        kf_idx2 = 2 * [next_kf_idxs[idx]]
        jvel_obs_line.set_xdata(idx2)
        gpos_obs_line.set_xdata(idx2)
        jvel_kf_line.set_xdata(kf_idx2)
        gpos_kf_line.set_xdata(kf_idx2)

    return animation.FuncAnimation(fig, viz_slice, repeat=False, frames=len(pairs), interval=300)


# Can't be used in one thread with matplotlib.backend == TkAgg
def viz_obs(obs: gcenv.Observation,
            scene_bounds: tuple[float, ...],
            scene_bins: int = 64,
            ) -> None:
    scene_bounds = np.asarray(scene_bounds)
    vgrid = VoxelGrid(scene_bounds=scene_bounds, nbins=scene_bins)
    voxels = vgrid.encode(obs, return_type='o3d')
    lb, ub = np.split(scene_bounds, 2)
    tcp_pos, tcp_rot, _ = np.split(obs.infer_action(), [3, 6])
    tcp_pos = (tcp_pos - lb) / (ub - lb)
    frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=tcp_pos)
    frame_tcp = frame_tcp.rotate(R.from_euler('ZYX', tcp_rot).as_matrix())
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([voxels, frame, frame_tcp])


if __name__ == '__main__':
    path = pathlib.Path(sys.argv[1]).resolve()
    ds = DemosDataset(path)
    gen = zip(ds, ds.as_demo_generator())
    for _ in range(60):
        _ = next(gen)
    try:
        while True:
            try:
                path, demo = next(gen)
                anim = viz_demo(demo)
            except StopIteration:
                break
            except AssertionError:
                print('Ill-formed demo ', path)
            except Exception as exc:
                print('Cant load demo ', path)
                raise exc
            else:
                plt.show()
                plt.close()
                obs = random.choice(demo)
                # (-0.7, -0.25, -0.03, -0.2, 0.25, 0.47)
                # (-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
                # viz_obs(obs, scene_bounds=(-0.7, -0.25, -0.1, -0.2, 0.25, 0.4), scene_bins=64)
    except KeyboardInterrupt:
        pass
