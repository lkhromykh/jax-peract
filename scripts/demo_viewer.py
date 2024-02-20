import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.embed_limit'] = 2 ** 20

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.dataset.dataset import DemosDataset
from src.dataset.keyframes_extraction import extractor_factory


def viz_demo(demo):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), squeeze=True, tight_layout=True)
    obs_ax, kf_ax, jvel_ax, gpos_ax = axes.flatten()
    pairs, kf_idxs = extractor_factory()(demo)

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
    jvel_ax.legend()
    gpos_ax.plot([obs.gripper_pos for obs in demo])
    gpos_ax.axhline(obs0.GRIPPER_OPEN_THRESHOLD, c='r', label='open_threshold')
    gpos_obs_line = gpos_ax.axvline(0, c='k')
    gpos_ax.legend()

    def viz_slice(idx):
        obs, kf = pairs[idx]
        obs_art.set_data(as_img(obs))
        kf_art.set_data(as_img(kf))
        idx2 = [idx, idx]
        jvel_obs_line.set_xdata(idx2)
        gpos_obs_line.set_xdata(idx2)

    return animation.FuncAnimation(fig, viz_slice, repeat=False, frames=len(pairs))


if __name__ == '__main__':
    path = sys.argv[1]
    path = os.path.join(os.path.dirname(__file__), path)
    path = os.path.abspath(path)
    ds = DemosDataset(path).as_demo_generator()
    idx = 0
    try:
        for idx, demo in enumerate(ds):
            anim = viz_demo(demo)
            plt.show()
    except Exception as exc:
        print('Bad demo idx ', idx)

