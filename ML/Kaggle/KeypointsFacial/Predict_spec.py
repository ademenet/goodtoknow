import numpy as np
import theano
import lasagne
import csv
import time
import sys
import json
from pandas import DataFrame
from pandas.io.parsers import read_csv
from Models import models

INDEXSTRING_30 = {'left_eye_center_x': 0, 'left_eye_center_y': 1,
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

INDEXSTRING_8 = {'left_eye_center_x': 0, 'left_eye_center_y': 1,
               'right_eye_center_x': 2, 'right_eye_center_y': 3, 
               'nose_tip_x': 4, 'nose_tip_y': 5,
               'mouth_center_bottom_lip_x': 6, 'mouth_center_bottom_lip_y': 7}

PATH = '/home/alain/Workspace/keyface/dataset/IdLookupTable.csv'


class Predict(object):
    def __init__(self, name, directory, size=30):
        self.output_size = size
        self.weight = directory + "/" + "w_" + name + ".npz"
        self.config = directory + "/" + name + ".json"

    def load_model(self):
        print "Load model..."
        with open('./save/' + self.config) as df:    
            self.config = json.load(df)
        self.network = models[self.config['model']](self.output_size)

    def load_weights(self):
        print "Load weights..."
        with np.load('./save/' + self.weight) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

    def load_inputs(self, cols=None):
        print "Load inputs..."
        fname = '~/Workspace/keyface/dataset/test.csv'
        df = read_csv(fname)
        df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
        if cols:
          df = df[list(cols) + ['Image']]
        df = df.dropna()
        self.X = np.vstack(df['Image'].values) / 255.
        self.X = self.X.reshape(-1, 1, 96, 96)
        self.X = self.X.astype(np.float32)
        return self.X

    def output(self):
        print "Predicting..."
        inputs = lasagne.layers.get_all_layers(self.network)[0].input_var
        pred = lasagne.layers.get_output(self.network, deterministic=True)
        get_pred = theano.function([inputs], pred)
        predict = get_pred(self.X)
        predict = predict * 48 + 48
        predict = predict.clip(0., 96.)
        return predict

def loop(values, group, df, nb):
    for index, row in group.iterrows():
        if nb == 30:
            values.append((df[int(row['ImageId']) - 1][INDEXSTRING_30[row['FeatureName']]]))
        elif nb == 8:
            values.append((df[int(row['ImageId']) - 1][INDEXSTRING_8[row['FeatureName']]]))            
    return values

def addrow(val):
    return np.arange(len(val))

output_30 = Predict(name=str(sys.argv[1]), directory=str(sys.argv[2]), size=30)
output_30.load_model()
output_30.load_weights()
output_30.load_inputs()
pred_30 = output_30.output()

output_8 = Predict(name=str(sys.argv[3]), directory=str(sys.argv[4]), size=8)
output_8.load_model()
output_8.load_weights()
output_8.load_inputs()
pred_8 = output_8.output()

reader = read_csv(PATH)
grouped = reader.groupby('ImageId')
values = []

for name, group in grouped:
    if len(group) > 8:
        loop(values, group, pred_30, 30)
    else:
        loop(values, group, pred_8, 8)

filename = 'submission_' + str(time.strftime('%y%m%d')) + '2.csv'
submission = DataFrame({'Location': values})
submission.index += 1
submission.to_csv(filename, index_label='RowId')
