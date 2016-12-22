import numpy as np
import theano
import lasagne
import csv
import time
import sys
import json
from pandas.io.parsers import read_csv
from Models import models

INDEXSTRING = {'left_eye_center_x': 0, 'left_eye_center_y': 1,
               'right_eye_center_x': 2, 'right_eye_center_y': 3,
               'left_eye_inner_corner_x': 4, 'left_eye_inner_corner_y': 5,
               'left_eye_outer_corner_x': 6, 'left_eye_outer_corner_y': 7,
               'right_eye_inner_corner_x': 8, 'right_eye_inner_corner_y': 9,
               'right_eye_outer_corner_x': 10, 'right_eye_outer_corner_y': 11,
               'left_eyebrow_inner_end_x': 12, 'left_eyebrow_inner_end_y': 13,
               'left_eyebrow_outer_end_x': 14, 'left_eyebrow_outer_end_y': 15,
               'right_eyebrow_inner_end_x': 16, 'right_eyebrow_inner_end_y': 17,
               'right_eyebrow_outer_end_x': 18, 'right_eyebrow_outer_end_y': 19,
               'nose_tip_x': 20, 'nose_tip_y': 21,
               'mouth_left_corner_x': 22, 'mouth_left_corner_y': 23,
               'mouth_right_corner_x': 24, 'mouth_right_corner_y': 25,
               'mouth_center_top_lip_x': 26, 'mouth_center_top_lip_y': 27,
               'mouth_center_bottom_lip_x': 28, 'mouth_center_bottom_lip_y': 29}

PATH = '/home/alain/Workspace/keyface/dataset/IdLookupTable.csv'

def load(cols=None):
  fname = '~/Workspace/keyface/dataset/test.csv'
  df = read_csv(fname)
  df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
  if cols:
      df = df[list(cols) + ['Image']]
  df = df.dropna()
  X = np.vstack(df['Image'].values) / 255.
  X = X.reshape(-1, 1, 96, 96)
  X = X.astype(np.float32)
  return X

def networkOutput(network, X_test):
  inputs = lasagne.layers.get_all_layers(network)[0].input_var
  pred = lasagne.layers.get_output(network, deterministic=True)
  get_pred = theano.function([inputs], pred)
  predicted = get_pred(X_test)
  return predicted

# Get file name
name = str(sys.argv[1])
directory = str(sys.argv[2])
output_size = int(sys.argv[3])
weight = directory + "/" + "w_" + name + ".npz"
config = directory + "/" + name + ".json"

# Load JSON
with open('./save/' + config) as data_file:    
  config = json.load(data_file)

# Load network
network = models[config['model']](output_size)

# Load weights
with np.load('./save/' + weight) as f:
    print len(f.files)
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

# Load inputs
X = load()

# Fit
output = networkOutput(network, X)
output = output * 48 + 48

with open(PATH, 'rb') as inp, open('submission_' + str(time.strftime('%y%m%d')) + "_" + name + '.csv', 'wb') as out:
    reader = csv.DictReader(inp)
    fields = ['RowId', 'Location']
    writer = csv.DictWriter(out, delimiter=',', fieldnames=fields)
    writer.writeheader()
    for row in reader:
        row['Location'] = min(output[int(row['ImageId']) - 1]
                              [INDEXSTRING[row['FeatureName']]], 96.)
        writer.writerow({'RowId': row['RowId'], 'Location': row['Location']})
