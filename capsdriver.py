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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

    for file in os.listdir(path):
        f = h5py.File(os.path.join(path, file), 'r')
        label = int(f.get('cjdata/label')[0][0])
        img = np.array(f.get('cjdata/image'))
        img = scipy.misc.imresize(img, (64, 64))
        X.append(img)
        l = [0, 0, 0]
        l[label - 1] = 1
        y.append(l)

    X = np.stack(X).reshape(len(X), 64, 64, 1).astype('float64') / 255
    y = np.stack(y)

    tts_split = train_test_split(
        X, y, range(y.shape[0]), test_size=0.3, random_state=0, stratify=y#np.argmax(y, axis=1)
    )
    
    x_train, x_test, y_train, y_test, train_idx, test_idx = tts_split
    
    x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
    x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)

    return x_train, x_test[:-300], y_train, y_test[:-300], X, y, x_test[-300:], y_test[-300:]


def load_tbi():
     # load data
    imgs = mri.load_imgs()

    match_df = mri.load_target('CT_Intracraniallesion_FIN')

    X, y = mri.match_image_ids(imgs, match_df)

    x_train, x_test, y_train, y_test = mri.get_split(X, y, 2)

    return x_train, x_test, y_train, y_test


def load_control():
    fn = '/vol/ml/ngetty/control/ABIDE/'
    #fn = '/Users/ngetty/Downloads/control/ABIDE/'
    ids = []
    X = []
    for root, dirnames, filenames in os.walk(fn):
        for filename in fnmatch.filter(filenames, '*.nii'):
            fn = os.path.join(root, filename)
            img = nib.load(fn)
            img_data = img.get_data()
            ids.append(filename[7:12])
            img_data = resize(img_data, (64, 64, 64))
            X.append(img_data)

    X = np.stack(X).reshape(len(X), 170, 256, 256, 1)

    infile = 'data/control.csv'
    df = pd.read_csv(infile, usecols=['Subject', 'Age'])
    df.set_index("Subject", drop=True, inplace=True)
    age_dict = df.to_dict(orient="index")

    y = np.array([age_dict[id] for id in ids])

    tts_split = train_test_split(
        X, y, range(y.shape[0]), test_size=0.2, random_state=0, stratify= y#np.argmax(y, axis=1)
    )

    x_train, x_test, y_train, y_test, train_idx, test_idx = tts_split

    x_train = x_train.reshape(x_train.shape[0], 170, 256, 256, 1) #.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 170, 256, 256, 1) #.astype('float32') / 255

    return x_train, x_test, y_train, y_test


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
    #x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(3, activation='sigmoid', name='pred')(x)
    model = Model(img_input, pred, name='mri_regressor')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cnn_model3D():
    img_input = Input(shape=(64, 64, 1), name='input')

    # --- block 1 ---
    x = Conv3D(64, (5, 5), activation='relu', padding='same', name='block1_conv1')(img_input)
    #x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Conv3D(64, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(800, activation='relu', name='fc_1')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(800, activation='relu', name='fc_2')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(3, activation='linear', name='pred')(x)
    model = Model(img_input, pred, name='mri_regressor')
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])

    return model


def main():
    args = params()
    d = 3
    if args.data == 'tumor':
        x_train, x_test, y_train, y_test, X, y, x_hold, y_hold = load_tumor()
        d = 2
    elif args.data == 'tbi':
        x_train, x_test, y_train, y_test = load_tbi()
    elif args.data == 'control':
        x_train, x_test, y_train, y_test = load_control()

    if args.sub > 0:
        x_train = x_train[:args.sub]
        y_train = y_train[:args.sub]

    print "Train freq:", np.array(Counter(np.argmax(y_train, axis=1)).values()).astype('float32') / len(y_train)
    print "Test freq:", np.array(Counter(np.argmax(y_test, axis=1)).values()).astype('float32') / len(y_test)
    print "Hold freq:", np.array(Counter(np.argmax(y_hold, axis=1)).values()).astype('float32') / len(y_hold)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    es = callbacks.EarlyStopping(min_delta=0.001, patience=10)
    lr_red = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    if args.cnn:
        if d == 2:
            c_model = cnn_model()
        else:
            c_model = cnn_model3D()
    if args.caps:
        model, eval_model, manipulate_model = capsnet.CapsNet(input_shape=x_train.shape[1:],
                                                              n_class=len(np.unique(np.argmax(y_train, 1))),
                                                              routings=args.routings, d=d)
        # compile the model
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

    if args.cnn:
        c_model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=args.verb,
                  callbacks=[lr_decay, es, lr_red],
                  validation_data=(x_test, y_test),
                  class_weight='auto')
        print c_model.evaluate(x_hold, y_hold)
    if args.caps:
        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[lr_decay, es, lr_red], verbose=args.verb)

        print model.evaluate([x_hold, y_hold], [y_hold, x_hold])


if __name__ == '__main__':
    main()





