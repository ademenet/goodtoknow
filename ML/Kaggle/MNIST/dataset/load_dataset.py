import numpy as np
import theano

def load_dataset():

    import gzip

    # As Yann Lecun says on his website (http://yann.lecun.com/exdb/mnist/):
    # "These files are not in any standard image format. You have to write your
    # own (very simple) program to read them. The file format is described at
    # the bottom of this page."
    def load_images(filename):
        with gzip.open(filename, mode="rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return datas / np.float32(256)

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz')
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print "Load datasets"
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
