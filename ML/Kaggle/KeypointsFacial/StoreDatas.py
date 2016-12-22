"""Utilities functions to store weight and historics of our networks.
"""

import numpy as np
import cPickle

class saveDatas(object):
    """
    """
    def __init__(self, filename, network):
        self.filename = filename
        self.network = network

    # def saveHistory()

    def dumpWeight(self)
        """Dump network's weight
        """
        np.savez(self.network + '.npz', *lasagne.layers.get_all_param_values(network))

    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
