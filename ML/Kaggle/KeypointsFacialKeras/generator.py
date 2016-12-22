import numpy as np
import skimage.transform as sk


class Generator(object):
    """Several useful methods in order to expand our dataset at train time and
    help generalize our model.

    Sources: http://cs231n.stanford.edu/reports2016/010_Report.pdf
    """

    def __init__(self,
                 X_train,
                 Y_train,
                 batchsize=32,
                 flip_ratio=0.5,
                 rotate_ratio=0.5,
                 contrast_ratio=0.5,
                 flip_indices=[(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]
                 ):
        """
        Arguments
        ---------
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.rotate_ratio = rotate_ratio
        self.contrast_ratio = contrast_ratio
        self.flip_indices = flip_indices

    # def mini_batch(self, inputs, targets, shuffle=False, batchsize=32):
    #     """Returns mini batches

    #     Solution took from Lasagne examples:
    #     https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

    #     Arguments
    #     ---------
    #         inputs:     Tensor4
    #         outputs:    fmatrix
    #         shuffle:    randomized indexes
    #         batchsize:  integer

    #     Returns
    #     -------
    #         Batchsize's arrays
    #     """
    #     assert len(inputs) == len(targets)
    #     if shuffle:
    #         indices = np.arange(len(inputs))  # Create as much as indices in inputs
    #         np.random.shuffle(indices)  # Shuffle them
    #     for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    #         if shuffle:
    #             excerpt = indices[start_idx:start_idx + batchsize]
    #         else:
    #             excerpt = slice(start_idx, start_idx + batchsize)
    #         yield inputs[excerpt], targets[excerpt]

    def _random_indices(self, ratio):
        """Generate random unique indices according to ratio"""
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    def flip(self):
        """Flip image batch"""
        indices = self._random_indices(self.flip_ratio)
        # if flip_indices == ():
        #     flip_indices = [ (0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11),
        #                     (12, 16), (13, 17), (14, 18), (15, 19), (22, 24),
        #                     (23, 25) ]
        self.inputs[indices] = self.inputs[indices, :, :, ::-1]
        self.targets[indices, ::2] = self.targets[indices, ::2] * -1
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = self.targets[indices, b], self.targets[indices, a]

    def rotate(self):
        """Rotate slighly the image and the targets. Work only with one channel"""
        indices = self._random_indices(self.rotate_ratio)
        angle = np.random.randint(-10, 10)
        for i in indices:
            self.inputs[i] = sk.rotate(self.inputs[i, 0, :, :], angle)
        angle = np.radians(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.targets = self.targets.reshape(len(self.targets), self.Y_train.shape[1] / 2, 2)
        self.targets[indices] = np.dot(self.targets[indices], R)
        self.targets = self.targets.reshape(len(self.targets), self.Y_train.shape[1])
        self.targets = np.clip(self.targets, -1, 1)

    def contrast(self):
        """Contrast jittering (reduction)"""
        indices = self._random_indices(self.contrast_ratio)
        delta = np.random.uniform(0.8, 1.2)
        self.inputs[indices] = (delta * self.inputs[indices, :, :, :]) + (1 - delta) * np.mean(self.inputs[indices, :, :, :])

    def generate(self, batchsize=32):
        """Generator"""
        while True:
            cuts = [(b, min(b + self.batchsize, self.size_train)) for b in range(0, self.size_train, self.batchsize)]
            for start, end in cuts:
                self.inputs = self.X_train[start:end].copy()
                self.targets = self.Y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]  # Need this to avoid indices out of bounds
                self.flip()
                self.rotate()
                self.contrast()
                yield (self.inputs, self.targets)
