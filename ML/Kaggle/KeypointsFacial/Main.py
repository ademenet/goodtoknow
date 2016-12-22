import os
import time
import json
import lasagne
import numpy as np
from Colors import bcolors
from LoadData import LoadData
from CNN import FitModel
from Models import models
# from sklearn.model_selection import ParameterGrid


# param_grid = {'reg_function': ['lasagne.regularization.l1', 'lasagne.regularization.l2'], 
#                 'reg_params': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}

# test = list(ParameterGrid(param_grid))   

# test = {
#     'net8-4t1': ((3, 3), (3, 3), (3, 3)), 
#     'net8-4t2': ((4, 4), (4, 4), (4, 4))
# }

config = {
          'date': time.strftime('%y%m%d'),
          'model': 'lenet5',
          'version': 'net10-1s-noseNmouth',
          'learning_rate': (0.03, 0.0001),
          'momentum': (0.9, 0.999),
          'decay': 'linear',
          'reg_params': 1e-5,
          'reg_function': 'lasagne.regularization.l2',
          'data_augmentation': (0.5, 0.5, 0.5),
          'max-pooling': ((2, 2), (2, 2), (2, 2)),
          'dropout': (0.3, 0.4, 0.5, 0.6),
          'epochs': 5000,
          'minibatchsize': 32,
          'update': 'nesterov_momentum',
          'w_init': None,
          'patience': 150,
          'ZCA': False,
          'comment': 'spec nose and mouth - net8-3s like with specific training on s mode',
          'time': 0,
          'mode': 's',
          }

if config['mode'] == 's':
    feature = ('left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y')
    flip_indices = [(0, 2), (1, 3)]
else:
    feature = ()
    flip_indices = [ (0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25) ]

datas = LoadData(mode=0)
X_t, X_v, y_t, y_v = datas.loadNSplit(feature=feature, spec=('nose_tip_x', 'nose_tip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'))
# Remove it when use normal cases
flip_indices = []

# ZCA whitening
if config['ZCA'] == True:
    from zca_whitening import zca_data
    X_t = zca_data(X_t, epsilon=1e-4)
    X_t = np.clip(X_t, -1, 1)
    X_v = zca_data(X_v, epsilon=1e-4)

print bcolors.BOLD + "Start training" + bcolors.ENDC

# for key, val in test.iteritems():
#     config['version'] = key
#     config['max-pooling'] = val

filename = str(config['date']) + '_' + str(config['model']) + '_' + str(config['version'])

print bcolors.HEADER + "--- Start training: " + filename + bcolors.ENDC

start = time.time()

network = models[config['model']](y_t.shape[1], config['dropout'], config['max-pooling'])

session = FitModel(network, filename, config)
session.initialization()
session.fit(X_t, X_v, y_t, y_v, flip_indices, config['patience'], save=100)

if not os.path.exists("./save/" + str(config['date'])):
    os.makedirs("./save/" + str(config['date']))
session.save_weights("save/" + str(config['date']))
session.save_loss(filename, "save/" + str(config['date']))

# Save and display end
config['time'] = time.time() - start
print "Training of " + str(filename) + " finished in " + str(config['time']) + " sec"

# Saving in config file:
with open("save/" + str(config['date']) + "/" + filename + '.json', 'w') as f:
    json.dump(config, f, indent=1)

print bcolors.BOLD + "=== THE END ===" + bcolors.ENDC
