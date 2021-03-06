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
import densenet


def params():
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on brain tumor classification.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--verb', default=1, type=int)
    parser.add_argument('--d', default=3, type=int)
    parser.add_argument('--dim', default=64, type=int)
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
    parser.add_argument('--dense', action='store_true',
                        help="Use dense model")
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


def load_tbi(dim, target='Age'):
    print "loading tbi data"
    imgs = mri.load_imgs(dim)

    #match_df = mri.load_target('CT_Intracraniallesion_FIN')

    match_df = mri.load_target(target)

    X, y = mri.match_image_ids(imgs, match_df)

    X = X.reshape(len(X), dim, dim, dim, 1)

    #mean = np.mean(y)
    #rnge = max(y) - min(y)
    #y = (y - mean) / rnge

    x_train, x_test, y_train, y_test = mri.get_split(X, y, 2, dim)

    return x_train, x_test, y_train, y_test


def load_control(d=3, dim=64):
    fn = '/vol/ml/ngetty/control/ABIDE/'
    #fn = '/Users/ngetty/Downloads/control/ABIDE/'
    ids = []
    X = []
    for root, dirnames, filenames in os.walk(fn):
        for filename in fnmatch.filter(filenames, '*.nii'):
            fn = os.path.join(root, filename)
            img = nib.load(fn)
            ids.append(int(filename[6:11]))
            img_data = img.get_data()
            if d ==3:
                img_data = resize(img_data, (dim, dim, dim))
            else:
                img_data = resize(img_data[:, :, 80], (dim, dim))
            X.append(img_data)

    if d == 3:
        X = np.stack(X).reshape(len(X), dim, dim, dim, 1)
    else:
        X = np.stack(X).reshape(len(X), dim, dim, 1)

    print "loading control target"
    infile = 'data/control.csv'
    df = pd.read_csv(infile, usecols=['Subject', 'Age', 'Sex', 'Description'])
    df = df.loc[df.Description == 'MP-RAGE']

    age_dict = dict(zip(df.Subject, df.Age))
    sex_dict = dict(zip(df.Subject, np.array(pd.get_dummies(df.Sex))))

    y = np.array([age_dict[id] for id in ids])
    sex_y = np.array([sex_dict[id] for id in ids])

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
        X, y, cat_y, sex_y, test_size=0.2, random_state=0, stratify=sex_y
    )

    x_train, x_test, y_train, y_test, bin_train, bin_test, sex_train, sex_test = tts_split

    if d == 3:
        x_train = x_train.reshape(x_train.shape[0], dim, dim, dim, 1).astype('float16')# / 255
        x_test = x_test.reshape(x_test.shape[0], dim, dim, dim, 1).astype('float16') #/ 255
    else:
        x_train = x_train.reshape(x_train.shape[0], dim, dim, 1).astype('float64')  # / 255
        x_test = x_test.reshape(x_test.shape[0], dim, dim, 1).astype('float64')  # / 255

    return x_train, x_test[:68], y_train, y_test[:68], x_test[68:], y_test[68:], mean, rnge, bin_train, bin_test[:68], bin_test[68:], sex_train, sex_test[:68], sex_test[68:]


def cnn_model(d=3, dim=64):

    w = 3
    if d == 3:
        conv = Conv3D
        pool = MaxPooling3D
        fil = (w,w,w)
        strides = (2,2,2)
    else:
        conv = Conv2D
        pool = MaxPooling2D
        fil = (w, w)
        strides = (2, 2)

    # --- block 1 ---
    img_input = Input(shape=(dim,) * d + (1,), name='input')

    # --- block 1 ---
    x = conv(dim, fil, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block1_pool')(x)
    x = conv(dim, fil, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block2_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(800, activation='relu', name='fc_1')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(800, activation='relu', name='fc_2')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(2, activation='softmax', name='pred')(x)
    model = Model(img_input, pred, name='mri_lesion')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cnn_model_age_sex(d=3, dim=64):
    w = 3

    if d == 3:
        conv = Conv3D
        pool = MaxPooling3D
        fil = (w,w,w)
        strides = (2,2,2)
    else:
        conv = Conv2D
        pool = MaxPooling2D
        fil = (w, w)
        strides = (2, 2)

    img_input = Input(shape=(dim,)*d + (1,), name='input')

    # --- block 1 ---
    '''x = conv(dim, fil, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block1_pool')(x)
    x = conv(dim, fil, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block2_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(800, activation='relu', name='fc_1')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(800, activation='relu', name='fc_2')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.8)(x)
    pred = Dense(1, activation='linear', name='pred')(x)
    sex_pred = Dense(2, activation='softmax', name='sex_pred')(x)'''


    # --- block 1 ---
    x = conv(64, fil, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = conv(64, fil, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block1_pool')(x)

    # --- block 2 ---
    x = conv(128, fil, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = conv(128, fil, activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block2_pool')(x)

    # --- coarse 1 branch ---
    c_1_bch = Flatten(name='c1_flatten')(x)
    c_1_bch = Dense(256, activation='relu', name='c1_fc_cifar10_1')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)
    c_1_bch = Dense(256, activation='relu', name='c1_fc2')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Dropout(0.5)(c_1_bch)
    sex_pred = Dense(2, activation='softmax', name='sex_pred')(c_1_bch)

    # --- block 3 ---
    x = conv(256, fil, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = conv(256, fil, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = conv(256, fil, activation='relu', padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = pool(strides, strides=strides, name='block3_pool')(x)

    # --- coarse 2 branch ---
    c_2_bch = Flatten(name='c2_flatten')(x)
    c_2_bch = Dense(1024, activation='relu', name='c2_fc_cifar100_1')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)
    c_2_bch = Dense(1024, activation='relu', name='c2_fc2')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Dropout(0.5)(c_2_bch)
    pred = Dense(1, activation='linear', name='pred')(c_2_bch)

    model = Model(img_input, [pred, sex_pred], name='mri_regressor')
    ls = 'mean_absolute_error'
    # ls = 'mean_squared_error'
    model.compile(loss=[ls, 'categorical_crossentropy'], optimizer='adam',
                  metrics={'pred': 'mae', 'sex_pred': 'accuracy'},
                  loss_weights=[0.75, 0.25])

    model.summary()
    return model


def unnorm(y, mean, rnge):
    #y = (y - np.mean(y)) / (max(y) - min(y))
    y = y * rnge + mean

    return y


def train_age_sex_cnn(model, x_train, y_train, x_test, y_test, x_hold, y_hold, sex_train, sex_test, sex_hold, args, calls):
    model.fit(x_train, [y_train, sex_train],
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=args.verb,
                callbacks=calls,
                validation_data=(x_test, [y_test, sex_test]))

    test_pred = model.predict(x_test, batch_size=10)
    hold_pred = model.predict(x_hold, batch_size=10)

    norm = 0
    if norm:
        tbi_pred = unnorm(tbi_pred, mean, rnge)
        test_pred = unnorm(test_pred, mean, rnge)
        hold_pred = unnorm(hold_pred, mean, rnge)

        y_tbi = unnorm(y_tbi, mean, rnge)
        y_test = unnorm(y_test, mean, rnge)
        y_hold = unnorm(y_hold, mean, rnge)

    print "Base Test:", mean_absolute_error(y_test, [np.mean(y_test)] * len(y_test))
    print "Base Hold:", mean_absolute_error(y_hold, [np.mean(y_hold)] * len(y_hold))

    tbi = 0

    if tbi:
        tbi_pred = model.predict(x_tbi, batch_size=10)
        print "Base TBI:", mean_absolute_error(y_tbi, [np.mean(y_tbi)] * len(y_tbi))
        print "TBI MAE:", mean_absolute_error(y_tbi, tbi_pred[0])
        print "TBI R2:", r2_score(y_tbi, tbi_pred[0])
        # print pd.DataFrame(zip(tbi_pred, y_tbi), columns=['y_pred', 'y_true'])

    print "Test MAE:", mean_absolute_error(y_test, test_pred[0])
    print "Hold MAE:", mean_absolute_error(y_hold, hold_pred[0])

    # print "Test R2:", r2_score(y_test, test_pred)
    # print "Hold R2:", r2_score(y_hold, hold_pred)

    print'Test acc:', np.sum(np.argmax(test_pred[1], 1) == np.argmax(sex_test, 1)) / float(y_test.shape[0])
    print'Hold acc:', np.sum(np.argmax(hold_pred[1], 1) == np.argmax(sex_hold, 1)) / float(y_hold.shape[0])

    print "Base Train Acc:", np.sum(np.argmax(sex_train, 1) == 1) / float(len(y_train))
    print "Base Test Acc:", np.sum(np.argmax(sex_test, 1) == 1) / float(len(y_test))
    print "Base Hold Acc:", np.sum(np.argmax(sex_hold, 1) == 1) / float(len(y_hold))

    # print pd.DataFrame(zip(test_pred[:,1], y_test), columns=['y_pred', 'y_true'])


def caps_sex_pred(model, x_train, y_train, x_test, y_test, x_hold, y_hold, args, calls):

    print "Base Train Acc:", np.sum(np.argmax(y_train, 1) == 1) / float(len(y_train))
    print "Base Test Acc:", np.sum(np.argmax(y_test, 1) == 1) / float(len(y_test))
    print "Base Hold Acc:", np.sum(np.argmax(y_hold, 1) == 1) / float(len(y_hold))

    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=calls, verbose=args.verb)

    print model.evaluate([x_test, y_test], [y_test, x_test], verbose=0)[3], \
    model.evaluate([x_hold, y_hold], [y_hold, x_hold], verbose=0)[3]

    y_pred, _ = model.predict(x_test, batch_size=15)
    print np.argmax(y_pred, 1), np.argmax(y_test, 1)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / float(y_test.shape[0]))

    y_pred, _ = model.predict(x_hold, batch_size=15)
    print('Hold acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_hold, 1)) / float(y_hold.shape[0]))


def pred_tbi_wcontrol(model, tbi_xtrain, tbi_ytrain, tbi_xtest, tbi_ytest, x_train, args, calls):
    y_train = np.array([[1, 0]] * len(x_train))

    x_train = np.concatenate([tbi_xtrain, x_train])
    y_train = np.concatenate([tbi_ytrain, y_train])

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=args.verb,
              callbacks=calls,
              validation_data=(tbi_xtest, tbi_ytest))


def main():
    args = params()
    d = args.d
    if args.data == 'tumor':
        x_train, x_test, y_train, y_test, x_hold, y_hold = load_tumor()
        m = 'val_acc'
        mo = 'max'
        classes = 3
    if args.data == 'tbi':
        x_train, x_test, y_train, y_test = load_tbi(args.dim, "CT_Intracraniallesion_FIN")
    if args.data == 'control':
        print "loading control"
        classes = 10
        x_train, x_test, y_train, y_test, x_hold, y_hold, mean, rnge, bin_train, bin_test, bin_hold, sex_train, sex_test, sex_hold = load_control(d, args.dim)
        print "Control train image shape:", x_train.shape
        tbi = False
        if tbi:
            tbi_xtrain, tbi_xtest, tbi_ytrain, tbi_ytest = load_tbi(args.dim, "CT_Intracraniallesion_FIN")
            print "Tbi train image shape:", tbi_xtrain.shape

        #m = 'val_pred_mean_absolute_error'
        #mo = 'min'
        m = 'val_acc'
        mo = 'max'
        #y_tbi = (y_tbi - mean) / rnge
        print "control loaded"


    if args.sub > 0:
        x_train = x_train[:args.sub]
        y_train = y_train[:args.sub]

    # print "Train freq:", np.array(Counter(np.argmax(y_train, axis=1)).values()).astype('float32') / len(y_train)
    # print "Test freq:", np.array(Counter(np.argmax(y_test, axis=1)).values()).astype('float32') / len(y_test)
    # print "Hold freq:", np.array(Counter(np.argmax(y_hold, axis=1)).values()).astype('float32') / len(y_hold)

    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    # es = callbacks.EarlyStopping(min_delta=0.001, patience=10, verbose=0)
    lr_red = callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    gb = GetBest(monitor=m, verbose=0, mode=mo)

    if args.dense:
        model = densenet.DenseNet(classes=3, input_shape=(64,64,1), depth=40, growth_rate=12,
                                  bottleneck=True, reduction=0.5)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=args.verb,
                  callbacks=[lr_decay, gb, lr_red],
                  validation_data=(x_test, y_test),
                  class_weight='auto')

        y_pred, _ = model.predict(x_test, batch_size=args.batch_size)
        y_pred = np.argmax(y_pred, 1)
        print('Test acc:', np.sum(y_pred == np.argmax(y_test, 1)) / float(y_test.shape[0]))

        y_hold_pred, _ = model.predict(x_hold, batch_size=args.batch_size)
        y_hold_pred = np.argmax(y_hold_pred, 1)
        print('Hold acc:', np.sum(y_hold_pred == np.argmax(y_hold, 1)) / float(y_hold.shape[0]))

    if args.cnn:
        if tbi:
            model = cnn_model(d, args.dim)
            pred_tbi_wcontrol(model, tbi_xtrain, tbi_ytrain, tbi_xtest, tbi_ytest, x_train, args, [lr_decay, lr_red, gb])
        else:
            model = cnn_model_age_sex(d, args.dim)
            train_age_sex_cnn(model, x_train, y_train, x_test, y_test, x_hold, y_hold, sex_train, sex_test, sex_hold,
                              args,
                              [lr_decay, lr_red, gb])
    if args.caps:
        model, eval_model, manipulate_model, reg_model = capsnet.CapsNet(input_shape=x_train.shape[1:],
                                                              n_class=2,
                                                              routings=args.routings, d=d)
        # compile the model
        model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

        '''eval_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=[capsnet.margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

        reg_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                      loss=['mse', 'mae'],
                      loss_weights=[args.lam_recon, 1.],
                      metrics={'reg': 'mae'})'''

        calls = [lr_decay, gb]
        caps_sex_pred(model, x_train, sex_train, x_test, sex_test, x_hold, sex_hold, args, calls)

    '''if args.cnn:
        c_model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=args.verb,
                  callbacks=[lr_decay, gb, lr_red],
                  validation_data=(x_test, y_test),
                  class_weight='auto')

        print c_model.evaluate(x_test, y_test, verbose=0)[1], c_model.evaluate(x_hold, y_hold, verbose=0)[1]'''


        #reg_model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
        #          validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[lr_decay, gb], verbose=args.verb)


if __name__ == '__main__':
    main()





