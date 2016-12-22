import time
import theano
import lasagne
from Utils import HistoryHandler
from Utils import EarlyStopping
from Utils import lr_mom_init
from DataAugmentation import DataAugmentation
import theano.tensor as T
import numpy as np
from Colors import bcolors


class FitModel(object):
    """Fit models define with Lasagne"""

    def __init__(self, network, name, learning_rate=(0.3, 0.001), momentum=(0.9, 0.999), epochs=500, minibatchsize=64, reg_params=0.0001):
        """Only two arguments are needed: network as neural network architecture
        and name, for saving datas. This approach allows us to iterate on differents
        architectures, save weight and metrics, compare and then pick up which one
        is the most performant

        Args:
            network: dictionnary, describe NN layers as lasagne do
            name: string, name of the actual NN architecture
            learning_rate: float
            momentum: float
            epochs: int
            batchsize: int
            history: object from HistoryHandler
        """
        self.network = network
        self.name = name
        self.learning_rate_start = learning_rate[0]
        self.learning_rate_end = learning_rate[1]
        self.momentum_start = momentum[0]
        self.momentum_end = momentum[1]
        self.epochs = epochs
        self.batchsize = minibatchsize
        self.reg_params = reg_params
        self.history = HistoryHandler(self.epochs)

    def mini_batch(self, inputs, targets, shuffle=False):
        """Returns mini batches

        Solution took from Lasagne examples:
        https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        Args:
            inputs: Tensor4
            outputs: fmatrix
            shuffle: randomized indexes

        Returns:
            Batchsize's arrays
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs)) # Create as much as indices in inputs
            np.random.shuffle(indices) # Shuffle them
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]

    def _monitoring_display(self, i, history, start_time, epochs=1):
        """Display indicators each period"""
        if i == 0:
            print " Epoch  |  Train loss  |  Valid loss  |   Accuracy   |   Time   "
            print "--------|--------------|--------------|--------------|----------"
        if i % epochs == 0:
            diff_time = time.time() - start_time
            print " %5.5d  |  %10.10s  |  %10.10s  |  %10.10s  |  in %.3f" % (i + 1, str(history['train_loss']), str(history['val_loss']), str(history['acc']), diff_time)

    def save_weights(self, features_name, dir):
        """Save our weights in .npz file"""
        weight_filename = dir + "/" + "w_" + str(self.name) + ".npz"
        np.savez(weight_filename, *lasagne.layers.get_all_param_values(self.network))

    def save_loss(self, features_name, dir):
        """Save loss results in files in order to plot later"""
        loss_filename = dir + "/" + str(self.name) + ".csv"
        np.savetxt(loss_filename, self.history.export_loss(), delimiter=",", fmt='%10f')

    def fit(self, X_train, X_val, y_train, y_val, patience=25, flip_indices=None):
        """Fit our model, whatever architecture.

        See descriptions to understand what is going on.
        """
        print bcolors.OKGREEN + "--- Start fitting!" + bcolors.ENDC

        # Initialize our variable in Theano object
        input_var = T.ftensor4('inputs')
        target_var = T.fmatrix('targets')
        # TODO: nouvelle target var car maintenant seulement vecteur

        # Get predictions thanks to this Lasagne method
        prediction = lasagne.layers.get_output(self.network, input_var)
        # faire predict validation

        # Regularization term
        reg = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)

        # We define the loss function using Mean Squarred Error
        loss = T.mean(lasagne.objectives.squared_error(target_var, prediction)) + self.reg_params * reg
        # tester RMSE

        # Get all trainable params
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # We make learning_rate and momentum as Theano shared variables so as to
        # decay them along epochs.
        lr = theano.shared(np.asarray(self.learning_rate_start, dtype=theano.config.floatX))
        mom = theano.shared(np.asarray(self.momentum_start, dtype=theano.config.floatX))
        lr_ = lr_mom_init(self.learning_rate_start, self.learning_rate_end, self.epochs, f='prog')        
        mom_ = lr_mom_init(self.momentum_start, self.momentum_end, self.epochs)

        # Update params using gradient descent and nesterov momentum
        # updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.03)
        updates = lasagne.updates.adam(loss, params)
        # updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    # learning_rate=lr, momentum=mom)
        # updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

        # Compute accuracy
        # accuracy = lasagne.objectives.squared_error(target_var, prediction).sum() / y_train.shape[1]
        accuracy = np.sqrt(T.mean(lasagne.objectives.squared_error(target_var, prediction))) * 48 # tester mathematiquement

        # Theano's functions: training and validation/test
        train_function = theano.function([input_var, target_var], loss, updates=updates)
        val_function = theano.function([input_var, target_var], [loss, accuracy])

        # Initializing early stop
        stop = EarlyStopping(patience=patience)

        # Start training - not more than number of epochs we define
        for epoch in range(self.epochs):
            start_time = time.time()

            # Iterate on mini batches for training
            _i = 0 # number of iteration
            _t = 0 # temporary loss value
            for X_t, y_t in self.mini_batch(X_train, y_train, shuffle=True):
                DataAugmentation(X_t, y_t).flip(flip_indices)
                # DataAugmentation(X_t, y_t).rotate()
                DataAugmentation(X_t, y_t).contrast()
                train_loss = train_function(X_t, y_t)
                _t += train_loss
                _i += 1
                self.history.update(epoch, train_loss)

            self.history.update(epoch, _t / _i)

            # Iterate on mini batches for testing
            _i = 0
            _t = 0
            _a = 0
            for X_v, y_v in self.mini_batch(X_val, y_val):
                # Wrong, I get an "SyntaxError: illegal expression for augmented assignment" for `val_loss, acc += val_function(X_v, y_v)``
                # I'd rather switch by this expression `x, y = (val - delta for val, delta in zip((x, y), (1, 2)))` from http://stackoverflow.com/questions/18132687/python-augmenting-multiple-variables-inline
                _t, _a = (old + new for old, new in zip((_t, _a), val_function(X_v, y_v)))
                _i += 1

            self.history.update(epoch, _t / _i, _a / _i, False)

            # Update hyper-parameters learning_rate and momentum
            lr.set_value(lr_[epoch])
            mom.set_value(mom_[epoch])

            # Monitoring training, validation loss and train/validation and time
            self._monitoring_display(epoch, self.history.dump(epoch), start_time)

            # Early stopping
            if stop.on_epoch(epoch, _t / _i):
                break
            # EarlyStopping(patience).__call__(epoch, self.network, _t / _i)
