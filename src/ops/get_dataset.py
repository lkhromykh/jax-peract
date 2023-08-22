import tensorflow as tf
import tensorflow_datasets as tfds
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass


def get_dataset(*,
                batch_size: int,
                img_size: tuple[int, int]
                ) -> tf.data.Dataset:
    ds = tfds.load('mnist', split='all', as_supervised=True)

    def normalize(image, label):
        image = tf.image.resize(image, img_size)
        return tf.cast(image, tf.float32) / 255., label
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(normalize)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.as_numpy_iterator()
