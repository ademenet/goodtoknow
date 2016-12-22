# Keras integration for Kaggle's Facial Keypoints Detection.
# Trying to beat Lasagne integration score 1.97116.
# This is made on educationnal purposes only. I aim to learn more
# about Keras and Theano.
#
# I set up my ~/.keras/keras.json as it:
# {
#     "image_dim_ordering": "th",
#     "epsilon": 1e-07,
#     "floatx": "float32",
#     "backend": "theano"
# }

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

import keras.backend as K

import numpy as np
from load_data import LoadData
from generator import Generator

# It is for reproductibility (np.random.rand will return allways the same
# random numbers)
# np.random.seed(1337)


def rmse(y_true, y_pred):
    """Compute the Root Mean Squared Error (RMSE)

    As custom metrics for Keras, the function need to take (y_true, y_pred) as parameters.
    As it is centered normalize, we multiply by 48 in order to get the real error.
    """
    rmse = np.sqrt(K.mean(K.square(y_pred - y_true))) * 48
    return rmse


# Define the name of the weights file that will be trained
weights_file_name = "test002.h5"

feature = ('left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y')
flip_indices = [(0, 2), (1, 3)]

# Load dataset using my previous LoadData class
load = LoadData()
X_train, X_val, Y_train, Y_val = load.loadNSplit(feature=feature)

# Define the output number
output_units = Y_train.shape[1]

# Define lenet5 like model
lenet5 = Sequential([
    Convolution2D(128, 3, 3, border_mode='valid', input_shape=(1, 96, 96)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Convolution2D(256, 2, 2),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Convolution2D(512, 2, 2),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(512),
    Activation('relu'),

    Dense(output_units),
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

print "> Compiling..."

checkpoint = ModelCheckpoint(weights_file_name, monitor='val_loss',
                             verbose=0, save_best_only=True, save_weights_only=True, mode='min')

earlystopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min')

lenet5.compile(loss='mean_squared_error', optimizer=sgd, metrics=[rmse])
lenet5.load_weights(weights_file_name)

gen_train = Generator(X_train,
                      Y_train,
                      batchsize=32,
                      flip_ratio=0.5,
                      rotate_ratio=0,
                      contrast_ratio=0.5,
                      flip_indices=flip_indices)

lenet5.fit_generator(gen_train.generate(), samples_per_epoch=gen_train.size_train, nb_epoch=500,
                     verbose=1, callbacks=[checkpoint, earlystopping], validation_data=(X_val, Y_val))

lenet5.save_weights(weights_file_name)
print "> Weights saved on disk as " + weights_file_name
