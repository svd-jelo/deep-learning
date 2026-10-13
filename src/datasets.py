import struct
import numpy as np

def load_examples_mnist(n_samples=1000,
                        images_path='data/mnist-handwritten-images/train-images.idx3-ubyte',
                        labels_path='data/mnist-handwritten-images/train-labels.idx1-ubyte',
                        random_state=None):
    """
    :param n_samples: number of samples loaded from the MNIST dataset.
    :param images_path: str; path to the image examples from the MNIST dataset.
    :param labels_path: str; path to the labels from the MNIST dataset.
    :param random_state: int; random seed
    :return: tuple; (X, y) where X is a 2d np.ndarray of shape (784, n_samples) representing the normalized features matrix and
    y is a 2d np.ndarray of shape (10, n_samples) representing the one-of-K encoded labels matrix.
    """

    with open(images_path,'rb') as f:
        _ , size = struct.unpack(">II", f.read(8))
        n_rows, n_cols = struct.unpack(">II", f.read(8))
        images = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        images = images.reshape((size, n_rows, n_cols))

    with open(labels_path,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = labels.reshape((size,))

    X_full = images.reshape((size,n_rows * n_cols))
    y_full = labels.reshape((size,1))
    data_combined = np.hstack((X_full, y_full))

    if not random_state:
        random_state = 42
    rng = np.random.default_rng(random_state)
    data_samp = rng.choice(data_combined, size=n_samples, replace=False, axis=0)
    X_samp, y_samp = data_samp[:,:-1], data_samp[:,-1]
    return X_samp, y_samp