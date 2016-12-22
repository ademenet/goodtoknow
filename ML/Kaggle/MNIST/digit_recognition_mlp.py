import numpy as np
import theano
import theano.tensor as T
import lasagne

def build_mlp(input_var=None):
    layer_input = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                            input_var=input_var)

    layer_hidden1 = lasagne.layers.DenseLayer(layer_input, num_units=800,
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.GlorotUniform())

    layer_hidden2 = lasagne.layers.DenseLayer(layer_hidden1, num_units=800,
                                              nonlinearity=lasagne.nonlinearities.rectify,
                                              W=lasagne.init.GlorotUniform())

    layer_out = lasagne.layers.DenseLayer(layer_hidden2, num_units=10,
                                            nonlinearity=lasagne.nonlinearities.softmax)

    return layer_out