import tensorflow as tf
import tensorflow_datasets as tfds
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass


def get_dataset(split: str,
                *,
                batch_size: int,
                img_size: tuple[int, int] | None = None
                ) -> tf.data.Dataset:
    ds, ds_info = tfds.load('cifar10', split=split,
                            as_supervised=True, with_info=True)

    def normalize(image, label):
        if img_size is not None:
            image = tf.image.resize(image, img_size)
        return tf.cast(image, tf.float32) / 255., label

    ds = ds.map(normalize)
    ds = ds.cache()
    if 'train' in split:
        ds = ds.repeat()
        drop_remainder = True
    else:
        drop_remainder = False
    ds = ds.shuffle(ds_info.splits[split].num_examples)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
