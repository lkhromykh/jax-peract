import pickle
import pathlib

from src.environment import gcenv
from src.environment.ur5_env import UREnv
from src.dataset.dataset import DemosDataset


def parse_demo(pkl_path: str) -> gcenv.Demo:
    with open(pkl_path, 'rb') as f:
        demo = pickle.load(f)
    return [UREnv.extract_observation(obs) for obs in demo]


def parse_dataset(from_dir: pathlib.Path) -> None:
    for dirname in from_dir.iterdir():
        new_dir = dirname.parent.parent / 'parsed_teleop' / dirname.name
        new_dir.mkdir(exist_ok=True, parents=True)
        ds = DemosDataset(new_dir)
        cont_idx = len(ds)
        for idx, path in enumerate(sorted(dirname.iterdir())):
            if idx < cont_idx:
                continue
            try:
                demo = parse_demo(path)
            except EOFError:
                continue
            else:
                ds.append(demo)
        print(f'{new_dir} size old {cont_idx} / new {len(ds)}')


if __name__ == '__main__':
    demos_root = pathlib.Path(fillme)
    parse_dataset(demos_root)

