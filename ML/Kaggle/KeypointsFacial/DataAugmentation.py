import numpy as np
import theano as T
import skimage.transform as sk


class DataAugmentation(object):
    """Several useful methods in order to expand our dataset at train time and
    help generalize our model.

    Sources: http://cs231n.stanford.edu/reports2016/010_Report.pdf
    """

    def __init__(self, config, flip_indices):
        """
        """
        self.flip_ratio = config[0]
        self.rotate_ratio = config[1]
        self.contrast_ratio = config[2]
        self.flip_indices = flip_indices

    def __call__(self, inputs, targets):
        self.inputs = inputs.copy()
        self.targets = targets.copy()
        self.y = targets.shape[1]
        self.flip()
        self.rotate()
        self.contrast()
        return self.inputs, self.targets

    def _random_indices(self, ratio):
        batchsize = self.inputs.shape[0]
        size = int(batchsize * ratio)
        return np.random.choice(batchsize, size, replace=False)

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
        angle = np.random.randint(-5, 5)
        for i in indices:
            self.inputs[i] = sk.rotate(self.inputs[i, 0, :, :], angle)
        angle = np.radians(angle)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.targets = self.targets.reshape(len(self.targets), self.y / 2, 2)
        self.targets[indices] = np.dot(self.targets[indices], R)
        self.targets = self.targets.reshape(len(self.targets), self.y)
        self.targets = np.clip(self.targets, -1, 1)

    def contrast(self):
        """Contrast jittering (reduction)"""
        indices = self._random_indices(self.contrast_ratio)
        delta = np.random.uniform(0.8, 1.2)
        self.inputs[indices] = (delta * self.inputs[indices, :, :, :]) + (1 - delta) * np.mean(self.inputs[indices, :, :, :])
