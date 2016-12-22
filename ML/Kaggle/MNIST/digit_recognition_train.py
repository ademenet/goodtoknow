import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
from digit_recognition_getdata import getData
from digit_recognition_mlp import build_mlp

# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
#     assert len(inputs) == len(targets)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batchsize]
#         else:
#             excerpt = slice(start_idx, start_idx + batchsize)
#         yield inputs[excerpt], targets[excerpt]

def train(epochs=100):

    # Load dataset
    print "Load dataset..."
    X_train, y_train, X_val, y_val, X_test, y_test = getData('train.csv', 'test.csv')

    # Initialize our theano's variables
    print "Initialize variables..."
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Select the network
    network = build_mlp(input_var)

    # Compute our predictions from the network
    prediction = lasagne.layers.get_output(network)

    # Then, compare the predictions with our target values
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # Do mean normalization
    loss = loss.mean()

    # Get parameters for our network
    params = lasagne.layers.get_all_params(network, trainable=True)

    # Update parameters using SGD with Nesterov momentum
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression but with deterministic as True which disabled
    # dropout
    # test_prediction = lasagne.layers.get_output(network)
    # test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    #                                                         target_var)
    # test_loss = test_loss.mean()

    # Compute accuracy
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                #   dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    test_fn = theano.function([input_var], prediction)

    # val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    # Start training
    print "Start training!"
    for epoch in range(epochs):
        print "Training epochs: %d / %d" % (epoch, epochs)
        train_err = 0
        # train_batches = 0
        # start_time = time.time()

        # for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            # inputs, targets = batch
            # train_err += train_fn(inputs, targets)
            # train_batches += 1

        inputs, targets = X_train, y_train
        train_err += train_fn(inputs, targets)

        # val_err = 0
        # val_acc = 0
        # val_batches = 0
        # for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        #     inputs, targets = batch
        #     err, acc = val_fn(inputs, targets)
        #     val_err += err
        #     val_acc += acc
        #     val_batches += 1

        # val_err, val_acc = val_fn(inputs, targets)

    print "Train error/loss: " + str(train_err)
    print "Expected:"
    print y_test[0]
    print "Learned:"
    print test_fn(X_test)[0]

    # print "Train accuracy: %d" % (test_acc)

    # test_err = 0
    # test_acc = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
    #     inputs, targets = batch
    #     err, acc = val_fn(inputs, targets)
    #     test_err += err
    #     test_acc += acc
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    # print("  test accuracy:\t\t{:.2f} %".format(
    #     test_acc / test_batches * 100))

train(500)