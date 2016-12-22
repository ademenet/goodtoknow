import numpy as np
import os
import time
from sklearn.cross_validation import train_test_split
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FILETRAIN = '~/Workspace/keyface/dataset/training.csv'
FILETEST = '~/Workspace/keyface/dataset/test.csv'

# function took from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
def load(test=False, cols=None):
    fname = FILETEST if test else FILETRAIN
    df = read_csv(os.path.expanduser(fname))
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if cols:
        df = df[list(cols) + ['Image']]

    # print(df.count())
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.reshape(-1, 1, 96, 96)
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_val, y_train, y_val


# print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
#     X.shape, X.min(), X.max()))
# print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
#     y.shape, y.min(), y.max()))

def monitoringDisplay(i, loss, val, acc, start_time, epochs=10):
    """Display indicators
    """
    if i == 0:
        print " Epoch  |  Train loss  |  Valid loss  |  Train / Val   "
        print "--------|--------------|--------------|----------------"
    if i % epochs == 0:
        print " %.5d  |  %.10s  |  %.10s  |  %.10s   |  in %.3f" % (i + 1, str(loss), str(val), str(acc), time.time() - start_time)


import theano
import theano.tensor as T
import lasagne

def build_mlp(input_var=None, depth=1, width=100):
    """Building custom mlp
    """
    network = lasagne.layers.InputLayer((None, 1, 96, 96), input_var)

    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network, num_units=width,
                                            nonlinearity=lasagne.nonlinearities.rectify,
                                            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.DenseLayer(network, num_units=30, nonlinearity=None)

    return network


max_epochs = 500

input_var = T.ftensor4('inputs')
target_var = T.fmatrix('targets')
X_train, X_val, y_train, y_val = load()

network = build_mlp(input_var)

prediction = lasagne.layers.get_output(network)
loss = T.mean(lasagne.objectives.squared_error(prediction, target_var))
# Ou bien pour loss:
# loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
# loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
# updates_sgd = updates.sgd(loss, params, learning_rate=0.01)
# updates = updates.apply_nesterov_momentum(updates_sgd, params, momentum=0.9)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001, momentum=0.9)

sqerr = lasagne.objectives.squared_error(y_val,prediction)
accuracy = np.sqrt(sqerr.sum()/30)

train_function = theano.function([input_var, target_var], loss, updates=updates)
val_function = theano.function([input_var, target_var], [loss, accuracy]) # ajouter un fonction accuracy


for e in range(max_epochs):
    train_error = 0
    start_time = time.time()
    train_error = train_function(X_train, y_train)
    val_error = 0
    val_error, val_acc = val_function(X_val, y_val)
    monitoringDisplay(e, train_error, val_error, val_acc, start_time, 1)

print "----- END -----"
