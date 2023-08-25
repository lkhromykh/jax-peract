import tensorflow as tf
import tensorflow_datasets as tfds
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass


def _mixup(ds: tf.data.Dataset, lambda_: float) -> tf.data.Dataset:
    sds = ds.shuffle(1000)
    ds = tf.data.Dataset.zip(sds, ds)
    def convex(x, y, lam): return (1. - lam) * x + lam * y

    def mixup(batch0, batch1):
        lam = tf.random.uniform((), 0, lambda_)
        return convex(batch0[0], batch1[0], lam),\
               convex(batch0[1], batch1[1], lam)
    return ds.map(mixup)


def get_dataset(split: str,
                *,
                batch_size: int,
                img_size: tuple[int, int] | None = None,
                mixup_lambda: float | None = None
                ) -> tf.data.Dataset:
    ds, ds_info = tfds.load('cifar10', split=split,
                            as_supervised=True, with_info=True)
    num_classes = ds_info.features['label'].num_classes

    def fn(image, label):
        if img_size is not None:
            image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.
        label = tf.one_hot(label, num_classes)
        return image, label

    ds = ds.map(fn)
    ds = ds.cache()
    if 'train' in split:
        ds = ds.repeat()
        if mixup_lambda is not None:
            ds = _mixup(ds, mixup_lambda)
        drop_remainder = True
    else:
        batch_size *= 4
        drop_remainder = False
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
