import numpy as np
import time
from pandas import DataFrame
from pandas.io.parsers import read_csv
from keras.models import load_model

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


def unnormalize(predict):
    predict = predict * 48 + 48
    predict = predict.clip(0., 96.)
    return predict


def load_inputs(cols=None):
    print "Load inputs..."
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


def loop(values, group, df, nb):
    for index, row in group.iterrows():
        if nb == 30:
            values.append((df[int(row['ImageId']) - 1]
                           [INDEXSTRING_30[row['FeatureName']]]))
        elif nb == 8:
            values.append((df[int(row['ImageId']) - 1]
                           [INDEXSTRING_8[row['FeatureName']]]))
    return values


X = load_inputs()

model_30 = load_model('lenet5.model')
model_30.load_weights('test001.h5')
pred_30 = model_30.predict(X)
pred_30 = unnormalize(pred_30)

model_8 = load_model('lenet5.model')
model_8.load_weights('test002.h5')
pred_8 = model_8.predict(X)
pred_8 = unnormalize(pred_8)

reader = read_csv(PATH)
grouped = reader.groupby('ImageId')
values = []

for name, group in grouped:
    if len(group) > 8:
        loop(values, group, pred_30, 30)
    else:
        loop(values, group, pred_8, 8)

filename = 'submission_' + str(time.strftime('%y%m%d')) + '.csv'
submission = DataFrame({'Location': values})
submission.index += 1
submission.to_csv(filename, index_label='RowId')
