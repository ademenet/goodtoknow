import time
import theano
import sys
import lasagne
import os
from Utils import HistoryHandler
from Utils import EarlyStopping
from Utils import lr_mom_init
from DataAugmentation import DataAugmentation
import theano.tensor as T
import numpy as np
from Colors import bcolors


class FitModel(object):
    """Fit models define with Lasagne"""

    def __init__(self, network, name, config):
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
        self.learning_rate_start = config['learning_rate'][0]
        self.learning_rate_end = config['learning_rate'][1]
        self.momentum_start = config['momentum'][0]
        self.momentum_end = config['momentum'][1]
        self.epochs = config['epochs']
        self.batchsize = config['minibatchsize']
        self.reg_params = config['reg_params']
        self.reg_function = config['reg_function']
        self.data_augmentation = config['data_augmentation']
        self.decay = config['decay']
        self.w_init = config['w_init']
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
            indices = np.arange(len(inputs))  # Create as much as indices in inputs
            np.random.shuffle(indices)  # Shuffle them
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
        if i % 100 == 0:
            print "... Training " + self.name + " in progress!"

    def save_weights(self, dir, weights=None):
        """Save our weights in .npz file"""
        if weights is not None:
            lasagne.layers.set_all_param_values(self.network, weights)
        date = time.strftime('%y%m%d')
        if not os.path.exists("./save/" + str(date)):
            os.makedirs("./save/" + str(date))
        weight_filename = dir + "/" + "w_" + str(self.name) + ".npz"
        np.savez(weight_filename, *lasagne.layers.get_all_param_values(self.network))

    def save_loss(self, features_name, dir):
        """Save loss results in files in order to plot later"""
        loss_filename = dir + "/" + str(self.name) + ".csv"
        np.savetxt(loss_filename, self.history.export_loss(), delimiter=",", fmt='%10f')

    def initialization(self):
        """Initialize before fitting process"""
        print bcolors.OKGREEN + "--- Initialization" + bcolors.ENDC

        # We make learning_rate and momentum as Theano shared variables so as to
        # decay them along epochs.
        self.lr = theano.shared(np.asarray(self.learning_rate_start, dtype=theano.config.floatX))
        self.mom = theano.shared(np.asarray(self.momentum_start, dtype=theano.config.floatX))
        self.lr_ = lr_mom_init(self.learning_rate_start, self.learning_rate_end, self.epochs, f=self.decay)
        self.mom_ = lr_mom_init(self.momentum_start, self.momentum_end, self.epochs, f=self.decay)

        # Initialize our variable in Theano object
        input_var = T.ftensor4('inputs')
        target_var = T.fmatrix('targets')

        # Get predictions thanks to this Lasagne method
        t_predict = lasagne.layers.get_output(self.network, input_var)
        v_predict = lasagne.layers.get_output(self.network, input_var, deterministic=True)

        # Regularization term
        exec('reg_method = ' + self.reg_function)
        reg = lasagne.regularization.regularize_network_params(self.network, reg_method)

        # We define the loss function using Mean Squarred Error
        t_loss = T.mean(lasagne.objectives.squared_error(target_var, t_predict)) + self.reg_params * reg
        v_loss = T.mean(lasagne.objectives.squared_error(target_var, v_predict)) + self.reg_params * reg
        # t_loss = np.sqrt(T.mean(lasagne.objectives.squared_error(target_var, t_predict))) + self.reg_params * reg
        # v_loss = np.sqrt(T.mean(lasagne.objectives.squared_error(target_var, v_predict))) + self.reg_params * reg
        # TODO: try with RMSE whereas MSE

        # Get all trainable params
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # Update params using gradient descent and nesterov momentum
        # updates = lasagne.updates.sgd(t_loss, params, learning_rate=self.lr)
        # updates = lasagne.updates.rmsprop(t_loss, params, learning_rate=0.03)
        # updates = lasagne.updates.adam(t_loss, params)
        updates = lasagne.updates.nesterov_momentum(t_loss, params,
                                                    learning_rate=self.lr, momentum=self.mom)

        # Compute accuracy
        # accuracy = lasagne.objectives.squared_error(target_var, prediction).sum() / y_train.shape[1]
        accuracy = np.sqrt(T.mean(np.square(target_var - v_predict))) * 48 # DONE: try math formula instead of method
        # accuracy = np.sqrt(t_loss) * 48 # DONE: try math formula instead of method
        # accuracy = np.sqrt(T.mean(lasagne.objectives.squared_error(target_var, t_predict))) * 48 

        # Theano's functions: training and validation/test
        self.train_function = theano.function([input_var, target_var], t_loss, updates=updates)
        self.val_function = theano.function([input_var, target_var], [v_loss, accuracy])

        if self.w_init is not None:
            w_file = 'save/' + str(self.w_init) + '.npz'
            with np.load(w_file) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)


    def fit(self, X_train, X_val, y_train, y_val, flip_indices, patience=0, save=0):
        """Fit our model, whatever architecture.

        See descriptions to understand what is going on.
        """
        print bcolors.OKGREEN + "--- Start fitting!" + bcolors.ENDC

        # Initializing early stop
        if patience > 0:
            stop = EarlyStopping(self.network, patience=patience)

        data = DataAugmentation(self.data_augmentation, flip_indices)

        # Start training - not more than number of epochs we define
        for epoch in range(self.epochs):
            start_time = time.time()

            X, y = data(X_train, y_train)

            # Iterate on mini batches for training
            _i = 0 # number of iteration
            _t = 0 # temporary loss value
            for X_t, y_t in self.mini_batch(X, y, shuffle=True):
                train_loss = self.train_function(X_t, y_t)
                _t += train_loss
                _i += 1

            self.history.update(epoch, _t / _i)

            # Iterate on mini batches for testing
            _i = 0
            _t = 0
            _a = 0
            for X_v, y_v in self.mini_batch(X_val, y_val):
                # Wrong, I get an "SyntaxError: illegal expression for augmented assignment" for `val_loss, acc += val_function(X_v, y_v)``
                # I'd rather switch by this expression `x, y = (val - delta for val, delta in zip((x, y), (1, 2)))` from http://stackoverflow.com/questions/18132687/python-augmenting-multiple-variables-inline
                _t, _a = (old + new for old, new in zip((_t, _a), self.val_function(X_v, y_v)))
                _i += 1

            self.history.update(epoch, _t / _i, _a / _i, False)

            # Update hyper-parameters learning_rate and momentum
            self.lr.set_value(self.lr_[epoch])
            self.mom.set_value(self.mom_[epoch])

            # Monitoring training, validation loss and train/validation and time
            self._monitoring_display(epoch, self.history.dump(epoch), start_time)

            # Early stopping
            if patience > 0 and stop.on_epoch(epoch, self.history.dump(epoch), olimit=0.65):
                self.save_weights(dir="save/" + str(time.strftime('%y%m%d')), weights=stop.bestWeights())
                break

            # Saving weights each n-epochs
            if save != 0 and epoch % save == 0:
                print bcolors.BOLD + "Saving weigths..." + bcolors.ENDC
                self.save_weights(dir="save/" + str(time.strftime('%y%m%d')))
            # EarlyStopping(patience).__call__(epoch, self.network, _t / _i)
