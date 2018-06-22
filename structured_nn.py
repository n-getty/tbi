#import pandas as pd
from fastai.structured import *
from fastai.column_data import *
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse


def params():
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on brain tumor classification.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")

    args = parser.parse_args()

    return args


infile = 'data/TRACKTBI_Pilot_DEID_02.22.18v2.csv'

df = pd.read_csv(infile, delimiter=",", na_values = ['', ' ', 'Untestable', 'QNS', 'NR', 'Unknown', 'Unk'])

# Drop the patient id
df = df.drop('PatientNum', axis=1)

df = df.replace('<', '', regex=True)
df = df.replace('>', '', regex=True)
df = df.infer_objects()


X = []
numX = df.select_dtypes(include=[np.number])
RS = 1
ED = 2
GCS = 2
PMH = 1
CT = 0
SES = 1
SHX = 1
defa = 1
bio = 1

if defa:
    default = df[['Age', 'Sex', 'RACE_3CAT']]
    X.append(default)

#ED test values
if ED:
    sw = 'ED'
    if ED > 1:
        sw = 'EDArr'
    edX = df[df.columns[pd.Series(df.columns).str.startswith(sw)]]
    X.append(edX)

#Glasgow Coma Scale values
if GCS:
    sw = 'gcs'
    if GCS > 1:
        sw = 'admgcs'
    gcsX = numX[numX.columns[pd.Series(numX.columns).str.lower().str.contains(sw)]]
    X.append(gcsX)

#CT Scan Labels
if CT:
    ctX = numX[numX.columns[pd.Series(numX.columns).str.startswith('CT')]]
    X.append(ctX)

#Previous Medical History
if PMH:
    pmhX = df[df.columns[pd.Series(df.columns).str.startswith('PMH')]]
    X.append(pmhX)

#Genetic snip columns
if RS:
    rsX = df[df.columns[pd.Series(df.columns).str.startswith('rs')]]
    X.append(rsX)

#Background
if SES:
    sesX = df[df.columns[pd.Series(df.columns).str.startswith('Ses')]]
    X.append(sesX)

#Drug use
if SHX:
    shxX = df[df.columns[pd.Series(df.columns).str.startswith('SHX')]]
    X.append(shxX)

#Biomarkers
if bio:
    bioX = df.iloc[:,726:813]
    X.append(bioX)

y = df['CT_Intracraniallesion_FIN']
y = to_categorical(y)
#y = pd.get_dummies(y)
#X.append(y)
df = pd.concat(X, axis=1)

#idxs = y.notnull()
#df = df.loc[idxs]

for c in df.select_dtypes(exclude=[np.number]):
    df[c] = pd.to_numeric(df[c], errors='ignore')

cat_vals = df.select_dtypes(exclude=[np.float])
cont_vals = df.select_dtypes(include=[np.float])

cat_vars = cat_vals.columns
contin_vars = cont_vals.columns

print(len(cat_vars))
print(len(contin_vars))

for v in cat_vars:
    df[v] = df[v].astype('category').cat.as_ordered()

for v in contin_vars:
    df[v] = df[v].astype('float64')

#cat_sz = [(c, len(df[c].cat.categories)+1) for c in cat_vals.columns]
#emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

df_sep, _, nas, _ = proc_df(df, do_scale=True)

'''train_ratio = 0.75
train_size = int(len(df_sep) * train_ratio)
val_idx = list(range(train_size, len(df_sep)))'''

tts_split = train_test_split(range(len(y)), test_size=0.2, random_state=0, stratify=y)

train_idx, val_idx = tts_split

cat_vals = df_sep.select_dtypes(include=[np.int8, np.int16])
cont_vals = df_sep.select_dtypes(include=[np.float64])

cat_sz = [(c, len(df[c].cat.categories)+1) for c in cat_vals.columns]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

cat_vars = cat_vals.columns
contin_vars = cont_vals.columns

print(len(cat_vars))

args = params()

md = ColumnarModelData.from_data_frame("test", val_idx, df_sep, y.astype(np.float32),
                                       cat_flds=cat_vars, is_reg=False, is_multi=True, bs=50)

learn = md.get_learner(emb_szs, cont_vals.shape[1], 0.4, 2, [500, 250, 125], [0.1, 0.2, 0.3])
learn.fit(1e-2, 30, cycle_len=1, cycle_mult=1)

y_pred = to_np(learn.model(to_gpu(V(T(np.array(cat_vals)))), to_gpu(V(T(np.array(cont_vals))))))

train_pred = y_pred[train_idx]
test_pred = y_pred[val_idx]

print("Base Train Acc:", np.sum(np.argmax(y[train_idx], 1) == 1) / float(len(y[train_idx])))
print("Base Test Acc:", np.sum(np.argmax(y[val_idx], 1) == 1) / float(len(y[val_idx])))

print('Train acc:', np.sum(np.argmax(train_pred, 1) == np.argmax(y[train_idx], 1)) / float(train_pred.shape[0]))
print('Test acc:', np.sum(np.argmax(test_pred, 1) == np.argmax(y[val_idx], 1)) / float(test_pred.shape[0]))