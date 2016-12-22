import numpy as np
import socket
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from Colors import bcolors


class LoadData:
    def __init__(self, test=False, mode=0):
        """
            mode:
                0: training
                1: small training dataset
                2: test
        """
        print bcolors.OKGREEN + "--- Loading datas..." + bcolors.ENDC
        self.test = test
        self.mode = mode
        self.LOCAL = ['~/Github/ML/Kaggle/KeypointsFacial/dataset/training.csv',
                      '~/Github/ML/Kaggle/KeypointsFacial/dataset/training_small.csv',
                      '~/Github/ML/Kaggle/KeypointsFacial/dataset/test.csv']

        self.REMOTE = ['~/Workspace/keyface/dataset/training.csv',
                       '~/Workspace/keyface/dataset/training_small.csv',
                       '~/Workspace/keyface/dataset/test.csv']
        self.file = self.LOCAL if socket.gethostname(
        ) == 'Alains-MacBook-Pro.local' else self.REMOTE

    def load(self, feature=None, spec=None):
        """Function from Daniel Nouri's tutorial on Deep Learning with Theano,
        Lasagne and NoLearn. It's a little bit pimped.

        Args:
            test: if True, load the test file (the one we need to predict).
                Otherwise load the training data.
            cols:

        Returns:
            Numpy array of inputs and outputs. If test is True, then it will
            return one array with "m" images. If not - test is False, it will
            return four arrays: two input/output arrays for training (~66% of
            the dataset) and two arrays for validation (~33%) randomly choosen.
        """
        fname = self.file[self.mode]
        datas = read_csv(fname)
        # Convert csv string values to numpy arrays:
        datas['Image'] = datas['Image'].apply(
            lambda img: np.fromstring(img, sep=' '))
        if feature:  # If we wan't to select one particular feature
            datas = datas[list(feature) + ['Image']]
        datas = datas.dropna()  # Remove rows with empty datas
        if spec:
            datas = datas[list(spec) + ['Image']]
        # vstack returns a ndarray (m, 9216) and we normalize: each values is
        # now between [0;1]
        X = np.vstack(datas['Image'].values) / 255.
        X = X.reshape(-1, 1, 96, 96)  # Reshape to fit lasagne variables
        X = X.astype(np.float32)  # Be sure type is float32
        if not self.test:
            y = datas[datas.columns[:-1]].values
            # Normalize, which means referential point is a horizontal line in the
            # center
            y = (y - 48) / 48
            y = y.astype(np.float32)
        else:
            y = None
        return X, y

    def loadNSplit(self, size=0.2, feature=None, spec=None):
        X, y = self.load(feature=feature, spec=spec)
        print bcolors.OKGREEN + "--- Splitting validation" + bcolors.ENDC
        # Split randomly dataset into t: train and v: validation datas
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=size)
        return X_t, X_v, y_t, y_v
