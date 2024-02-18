import pathlib
from typing import Any
import multiprocessing as mp

from src.config import Config
from src.logger import get_logger
from src.dataset.dataset import DemosDataset
from src.environment.rlbench_env import RLBenchEnv


def collect_task_demos(
        task: str,
        num_demos_per_task: int,
        dataset_dir: str,
        env_kwargs: dict[str, Any],
) -> DemosDataset:
    logger = get_logger()
    dataset_dir = pathlib.Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    dds = DemosDataset(dataset_dir)
    env = RLBenchEnv(**env_kwargs)
    env.TASKS = (task,)
    while len(dds) < num_demos_per_task:
        success = False
        while not success:
            desc = env.reset().observation.goal
            logger.info('Task: %s', desc)
            try:
                demo = env.get_demo()
            except Exception as exc:
                logger.error('Demo exception: %s', exc)
            else:
                logger.info('ep_length %d, total_eps %d', len(demo), len(dds))
                dds.append(demo)
                success = True
    env.close()
    return dds


if __name__ == '__main__':
    cfg = Config()

    def collect(task):
        return collect_task_demos(task=task,
                                  num_demos_per_task=cfg.num_demos_per_task,
                                  dataset_dir=pathlib.Path(cfg.datasets_dir).joinpath(task),
                                  env_kwargs=dict(scene_bounds=cfg.scene_bounds, time_limit=cfg.time_limit)
                                  )

    tasks = RLBenchEnv.TASKS
    with mp.Pool(processes=len(tasks)) as pool:
        pool.map(collect, tasks)
