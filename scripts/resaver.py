import os
import sys
import pickle
import pathlib
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.environment.ur5_env import UREnv
from src.dataset.dataset import DemosDataset
from src.logger import get_logger


def parse_task(to_dir: pathlib.Path, from_dir: pathlib.Path) -> None:
    task_dir = to_dir / from_dir.name
    task_dir.mkdir(exist_ok=True, parents=True)
    ds = DemosDataset(task_dir)
    for idx, demo_path in enumerate(sorted(from_dir.iterdir())):
        try:
            with open(demo_path, 'rb') as f:
                demo = pickle.load(f)
        except EOFError:
            get_logger().info('Cant read %s', demo_path)
            continue
        else:
            demo = list(map(UREnv.extract_observation, demo))
            demo[-2] = demo[-2].replace(is_terminal=False)
            ds.append(demo)
    get_logger().info('%s dataset new size: %d', task_dir, len(ds))


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from_path, to_path = map(lambda x: pathlib.Path(x).resolve(), sys.argv[1:3])
    fn = partial(parse_task, to_path)
    tasks = list(from_path.iterdir())
    with mp.Pool(processes=len(tasks)) as pool:
        pool.map(fn, tasks)
