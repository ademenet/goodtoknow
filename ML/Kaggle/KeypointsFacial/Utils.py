import numpy as np
from Colors import bcolors
import lasagne
import theano
import warnings


def lr_mom_init(start, stop, epochs, decay=0.06, f="linear"):
    if f=="linear":
        table = np.linspace(start, stop, epochs)
    elif f=="exp":
        table = start * np.exp(-decay * np.arange(epochs))
    elif f=="prog":
        table = start / (1. + (np.arange(epochs) * decay))
    return table.astype(theano.config.floatX)

# class EarlyStopping(object):
#     """EarlyStopping"""
#     def __init__(self, patience=25):
#         self.patience = patience
#         self.best_valid = np.inf
#         self.best_valid_epochs = 0
#         self.best_weights = None

#     def __call__(self, epoch, network, cur_valid_loss):
#         if cur_valid_loss < self.best_valid:
#             self.best_valid = cur_valid_loss
#             self.best_valid_epochs = epoch
#             self.best_weights = lasagne.layers.get_all_param_values(network)
#         if epoch > self.best_valid_epochs + self.patience:
#             print bcolors.OKBLUE + "Early stopping:"
#             print "best valid loss was " + str(self.best_valid) + " at epochs " + str(self.best_valid_epochs) + bcolors.ENDC
#             lasagne.layers.set_all_param_values(network, self.best_weights)
#             raise StopIteration()


class EarlyStopping(object):
    def __init__(self, network, delta=0, patience=0, mode='min'):
        self.network = network
        self.delta = delta
        self.patience = patience
        self.wait = 0
        self.min_delta = 1
        if mode not in ['min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown.' % (self.mode))
            mode = 'min'
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= 1
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= -1
            self.best = 0

    def on_epoch(self, epoch, history, olimit=0.65):
        overfitting = history['train_loss'] / history['val_loss']
        if self.monitor_op(history['val_loss'] - (self.min_delta * self.delta), self.best) and overfitting > olimit :
            self.best = history['val_loss']
            self.wait = epoch
            self.best_weights = lasagne.layers.get_all_param_values(self.network)      
            return False
        elif (epoch - self.wait) < self.patience:
            return False
        else:
            print bcolors.OKBLUE + "Early stopping:"
            print "--- Best valid loss was " + str(self.best) + " at epochs " + str(self.wait) + bcolors.ENDC
            return True

    def bestWeights(self):
        return self.best_weights


class HistoryHandler(object):
    """
    """
    def __init__(self, epochs):
        self.max_epochs = epochs
        self.train_loss = np.zeros(epochs)
        self.val_loss = np.zeros(epochs)
        self.acc = np.zeros(epochs)

    def update(self, epoch, loss_value, acc_value=None, train=True):
        if train:
            self.train_loss[epoch] = loss_value
        else:
            self.val_loss[epoch] = loss_value
            self.acc[epoch] = acc_value

    def dump(self, epoch):
        return dict(train_loss=self.train_loss[epoch],
                    val_loss=self.val_loss[epoch],
                    acc=self.acc[epoch])

    def export_loss(self):
        return (self.train_loss, self.val_loss, self.acc)