import os
import sys
import random
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
from src.utils.action_transform import DiscreteActionTransform
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
    jvel_ax.set_xticks(list(jvel_ax.get_xticks()) + kf_idxs)
    jvel_ax.set_xlim(left=0, right=kf_idxs[-1])
    jvel_ax.legend(loc='upper left')

    gpos_ax.plot([obs.gripper_pos for obs in demo])
    gpos_ax.axhline(obs0.GRIPPER_OPEN_THRESHOLD, c='r', label='open_threshold')
    gpos_obs_line = gpos_ax.axvline(0, c='k')
    gpos_kf_line = gpos_ax.axvline(kf_idxs[0], c='g')
    gpos_ax.set_xticks(list(gpos_ax.get_xticks()) + kf_idxs)
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


def viz_obs(obs: gcenv.Observation,
            scene_bounds: tuple[float, ...],
            scene_bins: int = 64,
            ) -> None:
    scene_bounds = np.asarray(scene_bounds)
    vgrid = VoxelGrid(scene_bounds=scene_bounds, nbins=scene_bins)
    act_trans = DiscreteActionTransform(scene_bounds, scene_bins, 72)
    breakpoint()
    act = obs.infer_action()
    act = act_trans.encode(act)
    act = act_trans.decode(act)
    voxels = vgrid.encode(obs, return_type='o3d')
    print('center ', voxels.get_center())
    print('max bound ', voxels.get_max_bound())
    print('min bound ', voxels.get_min_bound())
    print('bbox ', voxels.get_axis_aligned_bounding_box())
    act = obs.infer_action()
    frame_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=act[:3])# - scene_bounds[:3])
    frame_tcp = frame_tcp.rotate(R.from_euler('ZYX', act[3:6]).as_matrix())
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([voxels, frame, frame_tcp])


if __name__ == '__main__':
    path = sys.argv[1]
    path = os.path.join(os.path.dirname(__file__), '..', path)
    path = os.path.abspath(path)
    ds = enumerate(DemosDataset(path).as_demo_generator())
    idx = 0
    try:
        while True:
            try:
                idx, demo = next(ds)
                # anim = viz_demo(demo)
            except StopIteration:
                break
            except Exception as exc:
                print('Bad demo idx ', idx)
                raise exc
            else:
                # plt.show()
                # obs = random.choice(demo)
                obs = demo[0]
                #(-0.7, -.25, -.03, -.2, .25, 0.47)
                viz_obs(obs, scene_bounds=(-0.7, -0.25, -0.03, -0.2, 0.25, 0.47))
    except KeyboardInterrupt:
        pass
