import os
import sys
import pathlib
import multiprocessing as mp
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.config import Config
from src.builder import Builder
from src.logger import get_logger
from src.dataset.dataset import DemosDataset
from src.dataset.keyframes_extraction import extractor_factory

Path = str | pathlib.Path


def parse_dataset(builder: Builder, raw_ds_path: Path, processed_ds_path: Path) -> None:
    enc = builder.make_encoders()
    extract_fn = extractor_factory(observation_transform=enc.infer_state,
                                   keyframe_transform=enc.infer_action)
    ds = DemosDataset(raw_ds_path)
    tfds = ds.as_tf_dataset(extract_fn)
    get_logger().info('Saving dataset to %s', processed_ds_path)
    tfds.save(str(processed_ds_path))
    get_logger().info('Done saving %s', processed_ds_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    cfg = Config()
    builder = Builder(cfg)

    def fn(path):
        processed_path = builder.exp_path(Builder.DATASETS_DIR) / path.name
        return parse_dataset(builder, path, processed_path)
    datasets = list(pathlib.Path(cfg.datasets_dir).iterdir())
    with mp.Pool(processes=len(datasets)) as pool:
        pool.map(fn, datasets)
