import os
import sys
import pickle
import pathlib
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from peract.environment.ur5_env import UREnv
from peract.dataset.dataset import DemosDataset
from peract.logger import get_logger


def parse_demo(dataset_dir: pathlib.Path, demo_path: pathlib.Path) -> None:
    parsed_path = (dataset_dir / demo_path.parent.name / demo_path.stem).with_suffix('.npz')
    if parsed_path.exists():
        return None
    parsed_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        with demo_path.open(mode='rb') as f:
            demo = pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as exc:
        get_logger().info('Cant read %s: %s', demo_path, exc)
    else:
        demo = list(map(UREnv.extract_observation, demo))
        demo[-2] = demo[-2].replace(is_terminal=False)
        DemosDataset.save_demo(demo, parsed_path)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from_path, to_path = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:3])
    fn = partial(parse_demo, to_path)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(fn, from_path.rglob('*.pkl'))
