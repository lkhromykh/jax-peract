import tree
import numpy as np
from rlbench.demo import Demo
#parse_dataset :: Trajectory -> td.data.Dataset


def transform(demo: Demo) -> 'Trajectory':
    trajecoty = []
    for obs in demo:
        obs = transform(obs)
        trajecoty.append(obs)

    return tree.map_structure(lambda t: np.concatenate(*t), *trajecoty)


def keyframe_extraction(demo):
    ...
