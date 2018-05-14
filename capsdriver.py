import os
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import h5py
import capsnet
import mri
import nibabel as nib
import fnmatch
import pandas as pd
import argparse
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, MaxPooling3D,Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import callbacks, optimizers
from skimage.transform import resize
from collections import Counter
import tensorflow as tf
from GetBest import GetBest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import mean_absolute_error, r2_score
from collections import defaultdict


def params():
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on brain tumor classification.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--verb', default=1, type=int)
    parser.add_argument('--sub', default=0, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--cv', action='store_true',
                        help="Use cross validation")
    parser.add_argument('--cnn', action='store_true',
                        help="Use cnn model")
    parser.add_argument('--caps', action='store_true',
                        help="Use caps model")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--data', default='tumor')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    #print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def load_tumor():
    X = []
    y = []

    path = 'data/tumor_data'
    p_imgs = defaultdict(list)
    p_type = {}
    for file in os.listdir(path):
        f = h5py.File(os.path.join(path, file), 'r')
        label = int(f.get('cjdata/label')[0][0])
        p = f.get('cjdata/PID')
        pid = str(''.join([unichr(x[0]) for x in p]))
        img = np.array(f.get('cjdata/image'))
        img = scipy.misc.imresize(img, (64, 64))
        l = [0, 0, 0]
        l[label - 1] = 1
        p_imgs[pid].append(img)
        p_type[pid] = l

    X = p_type.keys()
    y = p_type.values()
    #X = np.stack(X).reshape(len(X), 64, 64, 1).astype('float64') / 255
    #y = np.stack(y)

    tts_split = train_test_split(
        X, y, range(len(y)), test_size=0.3, random_state=0, stratify=y#np.argmax(y, axis=1)
    )

    px_train, px_test, py_train, py_test, train_idx, test_idx = tts_split

    x_train, x_test, y_train, y_test = [],[],[],[]

    for id in px_train:
        x = p_imgs[id]
        y_train.extend([p_type[id]] * len(x))
        x = np.stack(x).reshape(len(x), 64, 64, 1).astype('float64') / 255
        x_train.extend(x)

    for id in px_test:
        x = p_imgs[id]
        y_test.extend([p_type[id]] * len(x))
        x = np.stack(x).reshape(len(x), 64, 64, 1).astype('float64') / 255
        x_test.extend(x)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
    x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test[:-300], y_train, y_test[:-300], x_test[-300:], y_test[-300:]


def load_tbi():
    # load data
    imgs = mri.load_imgs()

    #match_df = mri.load_target('CT_Intracraniallesion_FIN')

    match_df = mri.load_target('Age')

    X, y = mri.match_image_ids(imgs, match_df)

    X = X.reshape(len(X), 64, 64, 64, 1)

    mean = np.mean(y)
    rnge = max(y) - min(y)
    #y = (y - mean) / rnge

    #x_train, x_test, y_train, y_test = mri.get_split(X, y, 2)

    return X, y, mean, rnge


def load_control():
    fn = '/vol/ml/ngetty/control/ABIDE/'
    #fn = '/Users/ngetty/Downloads/control/ABIDE/'
    ids = []
    X = []
    dim = 32
    for root, dirnames, filenames in os.walk(fn):
        for filename in fnmatch.filter(filenames, '*.nii'):
            fn = os.path.join(root, filename)
            img = nib.load(fn)
            img_data = img.get_data()
            ids.append(int(filename[6:11]))
            img_data = resize(img_data, (dim, dim, dim))
            X.append(img_data)

    X = np.stack(X).reshape(len(X), dim, dim, dim, 1)

    infile = 'data/control.csv'
    df = pd.read_csv(infile, usecols=['Subject', 'Age', 'Description'])
    df = df.loc[df.Description == 'MP-RAGE']
    age_dict = dict(zip(df.Subject, df.Age))

    y = np.array([age_dict[id] for id in ids])

    mean = np.mean(y)
    rnge = max(y) - min(y)
    mi = min(y)
    bins = rnge / 10
    cat_y = []
    for v in y:
        cat = [0] * 10
        va = (v - mi - 1) / bins
        cat[int(va)] = 1
        cat_y.append(cat)

    #y = (y - mean) / rnge
    cat_y = np.array(cat_y)
    tts_split = train_test_split(
        X, y, cat_y, test_size=0.2, random_state=0
    )

    x_train, x_test, y_train, y_test, bin_train, bin_test = tts_split

    x_train = x_train.reshape(x_train.shape[0], dim, dim, dim, 1).astype('float16')# / 255
    x_test = x_test.reshape(x_test.shape[0], dim, dim, dim, 1).astype('float16') #/ 255

    return x_train, x_test[:68], y_train, y_test[:68], x_test[68:], y_test[68:], mean, rnge, bin_train, bin_test[:68], bin_test[68:]


def cnn_model():
    img_input = Input(shape=(64, 64, 1), name='input')

    # --- block 1 ---
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(800, activation='relu', name='fc_1')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(800, activation='relu', name='fc_2')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(3, activation='sigmoid', name='pred')(x)
    model = Model(img_input, pred, name='mri_regressor')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cnn_model3D():
    img_input = Input(shape=(64, 64, 64, 1), name='input')

    # --- block 1 ---
    x = Conv3D(64, (5,5,5), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    x = Conv3D(64, (5,5,5), activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(800, activation='relu', name='fc_1')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(800, activation='relu', name='fc_2')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(1, activation='linear', name='pred')(x)
    model = Model(img_input, pred, name='mri_regressor')
    ls = 'mean_absolute_error'
    #ls = 'mean_squared_error'
    model.compile(loss=ls, optimizer='adam', metrics=['mae'])
    model.summary()
    return model


def unnorm(y, mean, rnge):
    #y = (y - np.mean(y)) / (max(y) - min(y))
    y = y * rnge + mean

    return y


def main():
    args = params()
    d = 3
    if args.data == 'tumor':
        x_train, x_test, y_train, y_test, x_hold, y_hold = load_tumor()
        d = 2
        m = 'val_acc'
        mo = 'max'
        classes = 3
    if args.data == 'tbi':
        x_train, x_test, y_train, y_test, X, y = load_tbi()
    if args.data == 'control':
        classes = 10
        x_train, x_test, y_train, y_test, x_hold, y_hold, mean, rnge, bin_train, bin_test, bin_hold = load_control()
        x_tbi, y_tbi, tbi_mean, tbi_rnge = load_tbi()
        m = 'val_mean_absolute_error'
        mo = 'min'
        #y_tbi = (y_tbi - mean) / rnge

    if args.sub > 0:
        x_train = x_train[:args.sub]
        y_train = y_train[:args.sub]

    #print "Train freq:", np.array(Counter(np.argmax(y_train, axis=1)).values()).astype('float32') / len(y_train)
    #print "Test freq:", np.array(Counter(np.argmax(y_test, axis=1)).values()).astype('float32') / len(y_test)
    #print "Hold freq:", np.array(Counter(np.argmax(y_hold, axis=1)).values()).astype('float32') / len(y_hold)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    es = callbacks.EarlyStopping(min_delta=0.001, patience=10, verbose=0)
    lr_red = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    gb = GetBest(monitor=m, verbose=0, mode=mo)

    if args.cnn:
        if d == 2:
            c_model = cnn_model()
        else:
            c_model = cnn_model3D()
    if args.caps:
        model, eval_model, manipulate_model, reg_model = capsnet.CapsNet(input_shape=x_train.shape[1:],
                                                              n_class=classes,
                                                              routings=args.routings, d=d)
        # compile the model
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

        eval_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

        reg_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse', 'mae'],
                      loss_weights=[1., args.lam_recon, 1.],
                      metrics={'capsnet': 'accuracy', 'reg': 'mae'})

    if args.cnn:
        c_model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=args.verb,
                  callbacks=[lr_decay, gb, lr_red],
                  validation_data=(x_test, y_test),
                  class_weight='auto')

        if d ==3:
            tbi_pred = c_model.predict(x_tbi, batch_size=10)
            test_pred = c_model.predict(x_test, batch_size=10)
            hold_pred = c_model.predict(x_hold, batch_size=10)

            '''tbi_pred = unnorm(tbi_pred, mean, rnge)
            test_pred = unnorm(test_pred, mean, rnge)
            hold_pred = unnorm(hold_pred, mean, rnge)

            y_tbi = unnorm(y_tbi, mean, rnge)
            y_test = unnorm(y_test, mean, rnge)
            y_hold = unnorm(y_hold, mean, rnge)'''

            print "Base Test:", mean_absolute_error(y_test, [np.mean(y_test)] * len(y_test))
            print "Base Hold:", mean_absolute_error(y_hold, [np.mean(y_hold)] *len(y_hold))
            print "Base TBI:", mean_absolute_error(y_tbi, [np.mean(y_tbi)] * len(y_tbi))

            print "Test MAE:", mean_absolute_error(y_test, test_pred)
            print "Hold MAE:", mean_absolute_error(y_hold, hold_pred)
            print "TBI MAE:", mean_absolute_error(y_tbi, tbi_pred)

            print "Test R2:", r2_score(y_test, test_pred)
            print "Hold R2:", r2_score(y_hold, hold_pred)
            print "TBI R2:", r2_score(y_tbi, tbi_pred)

            print pd.DataFrame(zip(tbi_pred, y_tbi), columns=['y_pred', 'y_true'])
            print pd.DataFrame(zip(test_pred, y_test), columns=['y_pred', 'y_true'])
        else:
            print c_model.evaluate(x_test, y_test, verbose=0)[1], c_model.evaluate(x_hold, y_hold, verbose=0)[1]


    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    es = callbacks.EarlyStopping(min_delta=0.001, patience=10, verbose=0)
    lr_red = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    gb = GetBest(monitor='val_capsnet_acc', verbose=0, mode='max')

    if args.caps:
        if d == 3:
            reg_model.fit([x_train, bin_train], [bin_train, x_train, y_train], batch_size=args.batch_size, epochs=args.epochs,
                      validation_data=[[x_test, bin_test], [bin_test, x_test, y_test]], callbacks=[lr_decay, gb], verbose=args.verb)
        else:
            model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                      validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[lr_decay, gb, lr_red], verbose=args.verb)

            w = model.get_weights()
            print model.evaluate([x_test, y_test], [y_test, x_test], verbose=0)[3], model.evaluate([x_hold, y_hold], [y_hold, x_hold], verbose=0)[3]

            eval_model.set_weights(w)
            y_pred, _ = eval_model.predict(x_test, batch_size=100)
            print np.argmax(y_pred, 1), np.argmax(y_test, 1)
            print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / float(y_test.shape[0]))

            y_pred, _ = eval_model.predict(x_hold, batch_size=100)
            print('Hold acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_hold, 1)) / float(y_hold.shape[0]))


if __name__ == '__main__':
    main()





