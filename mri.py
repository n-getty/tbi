import os
import numpy as np
import nibabel as nib
from keras import metrics, optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D, Input
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers.normalization import BatchNormalization
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import argparse


def cnn_model(n_classes=1):
    '''Model design adapted from:
    Predicting brain age with deep learning from raw imaging data
    results in a reliable and heritable biomarker
    James H Cole et. al.
    https://arxiv.org/pdf/1612.02572.pdf'''
    print "Loading cnn model"
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

        if n_classes == 1:
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
        sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.03,)
        model.compile(loss=ls, optimizer=sgd, metrics=m)

    return model


def load_imgs():
    path = '/vol/ml/track-tbi/TBI-Data/TRACKPilotNiiData/'
    data = {}
    for subdir, dirs, files in os.walk(path):
        for name in dirs:
            p = os.path.join(path, name)
            fn = os.path.join(p, 'rsfmri.nii.gz')
            if os.path.isfile(fn):
                img = nib.load(fn)
                img_data = img.get_data()

                # Take last image
                #data[name[6:]] = img_data[:,:,:,-1]

                # Take aggregate image
                data[name[6:]] = np.mean(img_data, axis=3)

    print "Loaded %d images" % len(data)
    return data


def load_target(target):
    match_file = 'data/SF-GO_matchlist.csv'
    infile = 'data/TRACKTBI_Pilot_DEID_02.22.18v2.csv'
    df = pd.read_csv(infile, na_values = ['', ' ', 'Untestable', 'QNS', 'NR', 'Unknown', 'Unk'], usecols=['PatientNum', target])

    match_df = pd.read_csv(match_file, usecols=['GO', 'SF-####'])
    match_df = match_df[match_df['SF-####'].notnull()]

    targets = []
    for index, row in match_df.iterrows():
        sf = row['SF-####']
        sc = df.loc[df['PatientNum'] == sf, target].iloc[0]
        targets.append(sc)

    match_df['target'] = targets

    print "Patients with %s %d" % (target, sum(match_df['target'].notnull()))

    return match_df


def match_image_ids(imgs, match_df):
    keyset = set(imgs.keys())
    X = []

    idxs = []
    for i, id in enumerate(match_df['GO']):
        if str(id) in keyset:
            X.append(imgs[str(id)])
            idxs.append(i)

    X = np.array(X)

    print "Image shape:", X.shape

    print "Matched images"
    match_df = match_df.reset_index(drop=True)
    match_df = match_df.loc[idxs]

    print "Final number of matched patients with mri:", match_df.shape[0]
    print Counter(match_df['target'])

    y = np.array(pd.get_dummies(match_df['target']))

    return X, y


def get_split(X, y):
    tts_split = train_test_split(
        X, y, range(y.shape[0]), test_size=0.2, random_state=0, stratify=(y.shape[1] > 1) & y)

    X_train, X_test, y_train, y_test, train_idx, test_idx = tts_split

    print X_train.shape

    print X_train[0].shape

    X_train = X_train.reshape((X_train.shape[0], 64, 64, 35, 1))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 35, 1))

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test):
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

        print "Train data counts:", Counter(np.array(y_train)[:, 1])
        print "Test data counts:", Counter(np.array(y_test)[:, 1])

        model_path = 'model.h5'
        model.save(model_path)

        train_pred = model.predict(X_train)
        pred = model.predict(X_test)
        print pd.DataFrame(zip(y_test, pred))

        if y_train.shape[1]==1:
            print "Baseline train MAE:", mean_absolute_error(y_train, [np.mean(y_train)] * len(y_train))
            print "Baseline test MAE:", mean_absolute_error(y_test, [np.mean(y_test)] * len(y_test))

            print 'Train MAE: ', mean_absolute_error(y_train, train_pred)
            print 'Test MAE: ', mean_absolute_error(y_test, pred)
        else:
            print model.evaluate(X_train, y_train)
            print model.evaluate(X_test, y_test)


def get_parser():
    parser = argparse.ArgumentParser(description='Predict some TBI target using MRI images and CNN model')
    parser.add_argument("--target", default='CT_Intracraniallesion_FIN', type=str, help="What are you predicting?")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    imgs = load_imgs()

    match_df = load_target(args.target)

    X, y = match_image_ids(imgs, match_df)

    X_train, X_test, y_train, y_test = get_split(X, y)

    model = cnn_model(y.shape[1])

    train_model(model, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()