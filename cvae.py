import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 50


infile = 'data/TRACKTBI_Pilot_DEID_02.22.18v2.csv'

df = pd.read_csv(infile, delimiter=",", na_values = ['', ' ', 'Untestable', 'QNS', 'NR', 'Unknown', 'Unk'])
df = df.infer_objects()
# Drop the patient id
df = df.drop('PatientNum', axis=1)

df = df.replace(np.nan, -1)

for c in df.select_dtypes(exclude=[np.number]):
    df[c] = pd.to_numeric(df[c], errors='ignore')

df = df.select_dtypes(include=[np.number])

outcome_vars = ['GOSE', 'Neuro', 'Post', 'BSI', 'SWL', 'RPQ', 'CHARTS', 'TMT', 'PCL', 'PTSD', 'WAIS', 'CVLT', 'FIM']

#y = df['admGCS']
y = df[df.columns[pd.Series(df.columns).str.lower().str.contains('admgcs')]]


X = []
for pre in outcome_vars:
    tX = df[df.columns[pd.Series(df.columns).str.startswith(pre)]]
    X.append(tX)

X = pd.concat(X, axis=1)

tts_split = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
)

X_train, X_test, y_train, y_test = tts_split

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]

print("Outcome space:", n_x)
print("Condition space:", n_y)

# nubmer of epochs
n_epoch = 10

##  ENCODER ##

# encoder inputs
X = Input(shape=(n_x,))
cond = Input(shape=(n_y,))

# merge pixel representation and label
inputs = concatenate([X, cond])

# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu', name='dense_enc1')(inputs)
mu = Dense(n_z, activation='linear', name='dense_enc2')(h_q)
log_sigma = Dense(n_z, activation='linear', name='dense_enc3')(h_q)


def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(n_z,), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape=(n_z,))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##  DECODER  ##

# dense ReLU to sigmoid layers
decoder_hidden = Dense(512, activation='relu', name='dense_dec1')
decoder_out = Dense(n_x, activation='sigmoid', name='dense_dec_out')
h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model([X, cond], outputs)
encoder = Model([X, cond], mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z + n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)


# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl


def KL_loss(y_true, y_pred):
    return (0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))


def recon_loss(y_true, y_pred):
    return (K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics=[#KL_loss,
                                                      recon_loss])
cvae_hist = cvae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
                     validation_data=([X_test, y_test], X_test),
                     callbacks=[EarlyStopping(patience=5)])




