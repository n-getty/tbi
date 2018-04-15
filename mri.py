import os
import numpy as np
import nibabel as nib
from keras import metrics, optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D, Input
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers.normalization import BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, classification_report, roc_auc_score
import tensorflow as tf


def cnn_model(regr=False, n_classes=1):
    with tf.device('/gpu:1'):
        img_input = Input(shape=(64,64,35,1), name='input')
        
        m=2

        # --- block 1 ---
        x = Conv3D(8 * m, (3,3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = Conv3D(8 * m, (3,3,3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2,2,2), strides=(2,2,2), name='block1_pool')(x)

        # --- block 2 ---
        x = Conv3D(16 * m, (3, 3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(16 * m, (3, 3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)

        # --- block 3 ---
        x = Conv3D(32 * m, (3, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(32 * m, (3, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)

        # --- block 4 ---
        x = Conv3D(64 * m, (3, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(64 * m, (3, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)

        # --- block 5 ---
        x = Conv3D(128 * m, (3, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = BatchNormalization()(x)
        x = Conv3D(128 * m, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(x)

        if regr:
            ac = 'linear'
            ls = 'mean_absolute_error'
            m = [metrics.mae]
        else:
            ac = 'softmax'
            ls = 'categorical_crossentropy'
            #ls = 'binary_crossentropy'
            m = ['accuracy']

        # --- pred block ---
        x = Dropout(0.25)(x)
        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='fc')(x)
        x = Dropout(0.5)(x)
        pred = Dense(n_classes, activation=ac, name='pred')(x)

        # Compile model
        model = Model(img_input, pred, name='mri_regressor')
        sgd = optimizers.SGD(lr=0.001, momentum=0.9, #decay=0.03,
                             #nesterov=True,
                             )
        model.compile(loss=ls, optimizer=sgd, metrics=m)

    return model


def load_imgs():
    path = '/vol/ml/track-tbi/TBI-Data/TRACKPilotNiiData/'
    data = {}
    c = 0
    for subdir, dirs, files in os.walk(path):
        for name in dirs:
            p = os.path.join(path, name)
            fn = os.path.join(p, 'rsfmri.nii.gz')
            if os.path.isfile(fn):
                img = nib.load(fn)
                img_data = img.get_data()
                #data[name[6:]] = img_data[:,:,:,-1]
                data[name[6:]] = np.mean(img_data, axis=3)
                c+=1

    print "Loaded %d images" % c
    return data


target = 'CT_Intracraniallesion_FIN'

match_file = 'data/SF-GO_matchlist.csv'
infile = 'data/TRACKTBI_Pilot_DEID_02.22.18v2.csv'
df = pd.read_csv(infile, delimiter=",", na_values = ['', ' ', 'Untestable', 'QNS', 'NR', 'Unknown', 'Unk'])
df = df[['PatientNum', target]]

match_df = pd.read_csv(match_file)
match_df = match_df[match_df['SF-####'].notnull()]

scores = []
for index, row in match_df.iterrows():
    sf = row['SF-####']
    sc = df.loc[df['PatientNum'] == sf, target].iloc[0]
    scores.append(sc)

match_df['Scores'] = scores

#match_df = match_df[match_df['Scores'].notnull()]

print "Number of matched patients", match_df.shape[0]

imgs = load_imgs()
keyset = set(imgs.keys())
data = []

idxs = []
for i, id in enumerate(match_df['GO']):
    if str(id) in keyset:
        data.append(imgs[str(id)])
        idxs.append(i)

data = np.array(data)

print data.shape

print "Matched images"
match_df = match_df.reset_index(drop=True)
match_df = match_df.loc[idxs]

print "Final number of matched patients with mri:", match_df.shape[0]
print Counter(match_df['Scores'])

y = []
for sc in match_df['Scores']:
    if sc == 0:
        y.append([1,0])
    else:
        y.append([0,1])

tts_split = train_test_split(
    data, y, range(match_df.shape[0]), test_size=0.2, random_state=0, stratify=match_df['Scores'])

X_train, X_test, y_train, y_test, train_idx, test_idx = tts_split

print X_train.shape

print X_train[0].shape

X_train = X_train.reshape((X_train.shape[0], 64,64,35,1))
X_test = X_test.reshape((X_test.shape[0], 64,64,35,1))

print "Generated training split"

regr = False

n_classes = 2
model = cnn_model(regr, n_classes)

print "Loaded network model"

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger("results/mri_net.csv")

with tf.device('/gpu:1'):
    model.fit(X_train, y_train,
              batch_size=20,
              epochs=300,
              verbose=1,
              callbacks=[lr_reducer, early_stopper, csv_logger],
              validation_data=(X_test, y_test),
              class_weight='auto')

    print "Train data counts:", Counter(np.array(y_train)[:,1])
    print "Test data counts:", Counter(np.array(y_test)[:,1])

    model_path = 'model.h5'
    model.save(model_path)

    train_pred = model.predict(X_train)
    pred = model.predict(X_test)
    print pd.DataFrame(zip(y_test,pred))

    if regr:
        print "Baseline train MAE:", mean_absolute_error(y_train, [np.mean(y_train)] * len(y_train))
        print "Baseline test MAE:", mean_absolute_error(y_test, [np.mean(y_test)] * len(y_test))

        print 'Train MAE: ', mean_absolute_error(y_train, train_pred)
        print 'Test MAE: ', mean_absolute_error(y_test, pred)
    else:
        print model.evaluate(X_train, y_train)
        print model.evaluate(X_test, y_test)