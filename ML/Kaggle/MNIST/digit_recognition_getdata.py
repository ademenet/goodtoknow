import numpy as np

def getData(train_file, test_file):
    # Get training datas as numpy arrays
    train_datas = np.genfromtxt(train_file, delimiter=',', skip_header=1)
    y_train = train_datas[:, 0]
    # print y_train.shape
    # print y_train
    X_train = train_datas[:, 1:]
    X_train = X_train.reshape((len(X_train), 28, 28))
    X_train = X_train[:,np.newaxis,:,:]
    # print X_train.shape
    # We reserve the last 10000 training examples for validation.
    X_train, X_val, X_test= X_train[0:25200], X_train[25200:33600], X_train[33600:42000]
    y_train, y_val, y_test= y_train[0:25200], y_train[25200:33600], y_train[33600:42000]

    return X_train, y_train, X_val, y_val, X_test, y_test

# getData('train.csv', 'test.csv')