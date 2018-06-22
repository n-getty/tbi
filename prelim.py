import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error, classification_report, roc_auc_score, r2_score
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import copy
import matplotlib
matplotlib.matplotlib_fname()
pd.options.mode.chained_assignment = None
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from random import shuffle


infile = 'data/TRACKTBI_Pilot_DEID_02.22.18v2.csv'

df = pd.read_csv(infile, delimiter=",", na_values = ['', ' ', 'Untestable', 'QNS', 'NR', 'Unknown', 'Unk'])

# Drop the patient id
df = df.drop('PatientNum', axis=1)

df = df.replace('<', '', regex=True)
df = df.replace('>', '', regex=True)
df = df.infer_objects()


print "Original shape:", df.shape


def get_X(rs=0, ED=0, GCS=1, pmh=1, CT=0, SES=1, SHX=1, defa=1, bio=1, impute=0, encode=1, norm=1):
    X = []
    numX = df.select_dtypes(include=[np.number])

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
        gcsX = numX[numX.columns[pd.Series(numX.columns).str.lower().str.contains(sw)]]#& pd.Series(numX.columns).str.lower().str.contains('.t')]]
        X.append(gcsX)

    #CT Scan Labels
    if CT:
        ctX = numX[numX.columns[pd.Series(numX.columns).str.startswith('CT')]]
        X.append(ctX)

    #Previous Medical History
    if pmh:
        pmhX = df[df.columns[pd.Series(df.columns).str.startswith('PMH')]]
        X.append(pmhX)

    #Genetic snip columns
    if rs:
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

    X = pd.concat(X, axis=1)
    names = np.array(X.columns.tolist())

    for c in X.select_dtypes(exclude=[np.number]):
        X[c] = pd.to_numeric(X[c], errors='ignore')

    #mask = df.isnull().values

    if norm:
        for c in X.select_dtypes(include=[np.number]):
            X[c] = (X[c] - X[c].mean()) / (X[c].max() - X[c].min())

    if encode:
        for c in X.select_dtypes(exclude=[np.number]):
            X[c] = pd.get_dummies(X[c], dummy_na=True)

    if impute:
        imp = preprocessing.Imputer(missing_values=-1.0, strategy='mean', axis=0)
        X = imp.fit_transform(X)
        X = pd.DataFrame(X, columns=names)

    return X, names


def plot(X, names):
    #Correlation of numerical features
    corr = X.corr()

    c = corr.abs()
    s = c.unstack()
    s = s[s!=1.0]
    idxs = range(0, len(s), 2)
    so = s.sort_values()[::-1]

    print so[idxs][:20]

    '''corr = X.corr()
        plt.matshow(corr)
        plt.show()'''
    #f, ax = plt.subplots(figsize=(10, 8))
    #sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #            square=True, ax=ax)
    #plt.show()


def rand_jitter(arr):
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def cluster(X, df, name, m):
    y=df
    #model = RandomTreesEmbedding()

    if m =='tsne':
        model = TSNE(n_components=2, verbose=2, n_iter=1000, random_state=42)
        tsne_coords = model.fit_transform(X)
    elif m=='pca':
        pca = PCA(n_components=2, iterated_power=1000, random_state=42)
        tsne_coords = pca.fit_transform(X)

    #ns = ['Site', 'Race', 'Sex', 'TBILoc', 'HospDisch', 'PostPatientOutcome_12mo', 'PostRehab_12mo', 'PTSD_DSMIV_12mo', 'NeuroProcedureInHospital', 'PatientTypeCoded', 'GCSMildModSevereRecode']
    ns = ['GCSMildModSevereRecode', 'HospDisch', 'GOSE_OverallScore3M']
    tsne_coords[:,0] = rand_jitter(tsne_coords[:,0])
    tsne_coords[:, 1] = rand_jitter(tsne_coords[:, 1])
    colors = ['r', 'b', 'g', 'y', 'k']
    for n in ns:
        uniq = set(y[n])
        if np.nan in uniq:
            uniq.remove(np.nan)
        fig, ax = plt.subplots()
        c = 0
        scs = []
        df = pd.DataFrame({'x': tsne_coords[:, 0], 'y': tsne_coords[:, 1], 'l': y[n]})
        if len(uniq) < 6:
            for s in uniq:
                sdf = df[df.l == s]
                scs.append(ax.scatter(sdf.x, sdf.y, c=colors[c], marker='.'))
                c += 1
            plt.legend(scs, uniq)
        else:
            sc = ax.scatter(df.x, df.y, c=df.l, cmap="cool", marker='.')
            cbar = fig.colorbar(sc, ax=ax)
            #cbar.ax.set_ylabel(n, rotation=270)

        ax.set_title(n)
        # plt.show()
        fig.savefig(name + n[:3])

    kmeans = KMeans(n_clusters=5, random_state=0).fit(tsne_coords)

    #ccount = Counter(kmeans.labels_)

    clabels = kmeans.labels_

    cluster_map = pd.DataFrame(zip(clabels, y[ns[0]], y[ns[2]]), columns=['Cluster', 'Severity', 'Outcome'])

    m1_idxs = []
    m1_sev = 0
    m2_sev = 0
    m2_idxs = []
    for c in range(5):
            cluster = cluster_map[cluster_map['Cluster'] == c]
            #idx = cluster.index.tolist()
            #t = cluster['Severity'].str.startswith('S')
            #sevc = np.sum(t)

            '''if sevc > m1_sev:
                m2_idxs = m1_idxs
                m2_sev = m1_sev

                m1_idxs = idx
                m1_sev = sevc

            elif sevc > m2_sev:
                m2_idxs = idx
                m2_sev = sevc'''

            #cluster_count = Counter(zip(cluster['Severity'], cluster['Outcome']))
            print "Cluster %d stdev:" % c
            print np.std(cluster.Outcome)


    m1_idxs.extend(m2_idxs)
    X['Cluster'] = cluster_map['Cluster']
    X['HospDisch'] = y['HospDisch']
    sevX = X.loc[m1_idxs]
    sevX = sevX[sevX['admGCS'] < 13]
    sevX.to_csv('data/severe_cluster_data.csv')

    return sevX


def loo(model, col):
    idxs = df[col].notnull()
    y = df[col][idxs]
    print 'Non-null y:', len(y)
    global X

    X = np.array(X[idxs])

    loo = LeaveOneOut()
    loo.get_n_splits(X)
    y_pred = []

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        y_pred.append(p)

    print col
    print "Fmeasure:", f1_score(y, y_pred, average=None)
    print "Precision:", accuracy_score(y, y_pred)
    print "ROC:", roc_auc_score(y, y_pred)


def predict_missing(model, X):
    offs = []
    maes = []
    all = X.copy()
    for col in all:
        X = X.drop(col, axis=1)
        y = all[col]
        idxs = y.notnull()
        y = y[idxs]

        print 'Non-null y:', len(y)

        model.fit(X[idxs], y)

        y_pred = cross_val_predict(model, X[idxs], y, cv=5)

        mae = mean_absolute_error(y, y_pred)
        off = mae - mean_absolute_error(y, [np.mean(y)] * len(y))

        print col
        print "MAE:", mae
        print "Off-baseline:", off
        offs.append(off)
        maes.append(mae)
        X = all.copy()

    offdf = pd.DataFrame(zip(all.columns.get_values(), offs, maes), columns=['Label', 'Error', 'MAE'])
    offdf.sort_values(by='Error', inplace=True)
    print offdf


def predict_one(model, col):
    idxs = df[col].notnull()
    y = df[col][idxs]

    if col == 'GOSE_OverallScore3M':
        y[y<7] = 0
        y[y>6] = 1
        '''tmp_idx = y<5
        y[y<7] = 1
        y[y>1] = 2
        y[tmp_idx] = 0'''

    #y = y.astype('int')

    print 'Non-null y:', len(y)

    print col
    print
    print y.value_counts() / len(y)
    print

    if y.dtype == np.int64:
        tn = ['Yes', 'No']
    else:
        tn = y.unique()
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)

    model.fit(X[idxs],y)
    '''coef = np.array(model.coef_)
    sort_idx = np.argsort(coef)[0][::-1]
    names = np.array(num_names)
    print zip(names[sort_idx], coef[0][sort_idx])'''

    feat_score = model.feature_importances_
    #feat_score = model.coef_
    feat_idx = np.argsort(feat_score)[::-1]

    y_pred = cross_val_predict(model, X[idxs], y, cv=5)

    #print(classification_report(y, y_pred, target_names=tn))
    print "Fmeasure:", f1_score(y, y_pred, average=None)
    print "Precision:", accuracy_score(y, y_pred)
    if len(tn) == 2:
        print "ROC:", roc_auc_score(y, y_pred)
    print pd.DataFrame(zip(np.array(num_names)[feat_idx], feat_score[feat_idx])).head(20)
    return num_names[feat_idx]


def unnorm(y, mean, rnge):
    #y = (y - np.mean(y)) / (max(y) - min(y))
    y = y * rnge + mean

    return y


def reg_one(model, col):
    idxs = df[col].notnull()
    y = df[col][idxs]
    print 'Non-null y:', len(y)

    mean = np.mean(y)
    rnge = max(y) - min(y)
    y = (y - mean) / rnge

    model.fit(X[idxs],y)

    feat_score = model.feature_importances_
    feat_idx = np.argsort(feat_score)[::-1]

    y_pred = cross_val_predict(model, X[idxs], y, cv=5)

    y = unnorm(y, mean, rnge)
    y_pred = unnorm(y_pred, mean, rnge)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    #pred_df = pd.DataFrame({'y': y, 'pred': y_pred})

    #pred_df.sort_values(by='y', axis=0, inplace=True)

    #pred_df.plot(x='y', y='pred', kind='scatter', style='.')

    '''fig, ax = plt.subplots()
    scs = []
    scs.append(ax.scatter(range(len(y_pred)), y_pred, c='r', marker='.'))
    scs.append(ax.scatter(range(len(y)), y, c='b', marker='.'))
    plt.legend(scs, ['Pred', 'Truth'])

    ax.set_title(col)'''
    plt.show()

    print col
    print
    print y.value_counts() / len(y)
    print
    print "MSE:", mse
    print "MAE:", mae
    print "R2:", r2_score(y, y_pred)
    print "Baseline MAE", mean_absolute_error(y, [np.mean(y)] * len(y))
    print "Baseline MAE", mean_squared_error(y, [np.mean(y)] * len(y))
    #print pd.DataFrame(zip(np.array(num_names)[feat_idx], feat_score[feat_idx])).head(20)
    return num_names[feat_idx]


def predict_all(model):
    freq_t = 5
    res = []
    for n in label_names:
        idxs = df[n].notnull()
        y = df[n][idxs]
        vc = y.value_counts().values
        if len(y) > 100 and len(y.unique()) > 1 and vc[-1] > freq_t:
            print n
            y_pred = cross_val_predict(model, X[idxs], y, cv=2)
            f1 = f1_score(y, y_pred, average='macro')
            res.append((n, f1))

    res.sort(key=lambda x: x[1], reverse=True)

    print pd.DataFrame(res, columns=['Label', 'Macro F1']).head(50)


def all_feat(model):
    freq_t = 5
    feat_scores = np.zeros(X.shape[1])
    scaler = MinMaxScaler()
    for n in label_names:
        idxs = df[n].notnull()
        y = df[n][idxs]
        vc = y.value_counts().values
        if len(y) > 100 and len(y.unique()) > 1 and vc[-1] > freq_t:
            print n
            m = copy.copy(model)
            m.fit(X[idxs], y)
            feat_score = np.ravel(scaler.fit_transform(m.feature_importances_.reshape(-1, 1)))
            np.add(feat_scores, feat_score, out=feat_scores)

        sorted_feats = np.argsort(feat_scores)[::-1]

    print "Top 25 feats:", num_names[sorted_feats[:50]]


def top_cluster(col):
    global X, num_names
    allX, num_names = get_X(0, 1, 1, 1, 1, 1,0,1,1,1)

    X, num_names = get_X(0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    top_feats_pmh = predict_one(model, col)[:15]

    X, num_names = get_X(0, 0, 2, 0, 0, 0,0,0,0,1)
    #X = (X - X.mean()) / (X.max() - X.min())
    top_feats_ed = predict_one(model, col)[:5]

    #X, num_names = get_X(0, 1, 0, 0, 0, 0,0,0,0,1)
    #top_feats_rs = predict_one(model, col)[:5]

    X, num_names = get_X(0, 0, 0, 0, 0, 1, 0, 0, 0, 1)
    top_feats_ct = predict_one(model, col)[:15]

    X, num_names = get_X(0, 0, 0, 0, 0, 0, 0, 0, 1, 1)
    top_feats_demo = predict_one(model, col)[:2]

    X, num_names = get_X(0, 0, 0, 2, 0, 0, 0, 0, 0, 1)
    gcs = predict_one(model, col)

    X, num_names = get_X(0, 0, 0, 0, 0, 0, 0, 1, 0, 1)
    shx = predict_one(model, col)[:-1]

    top_feats = np.concatenate([top_feats_pmh, top_feats_ct, top_feats_demo, shx, gcs])
    cluster(allX[top_feats], df, 'figs/tsne_top_', 'tsne')


def private(model, col, n_teachers):
    idxs = np.array(range(len(X)))
    shuffle(idxs)
    student = copy.copy(model)
    teachers = [copy.copy(model)] * (n_teachers - 1)
    y = df[col][idxs]
    shuffX = X.loc[idxs]

    sub = len(X)/n_teachers
    stud_x = shuffX[-sub:-sub/2]
    stud_y = y[-sub:-sub/2]
    t_preds = []
    for x in range(n_teachers-1):
        sub_x = shuffX[x*sub: (x+1)*sub]
        sub_y = y[x*sub: (x+1)*sub]
        m = teachers[x]
        m.fit(sub_x, sub_y)
        t_preds.append(m.predict(stud_x))

    t_preds = np.stack(t_preds, axis=1)

    #preds = []
    #for row in t_preds:
    #    preds.append(np.bincount(row).argmax())

    test_y = y[:-sub]
    test_x = shuffX[:-sub]
    val_y = y[-sub / 2:]
    val_x = shuffX[-sub / 2:]
    student.fit(stud_x, stud_y)
    test_preds = student.predict(test_x)
    val_preds = student.predict(val_x)
    #print "Train ROC:", roc_auc_score(stud_y, preds)
    #print "Test ROC:", roc_auc_score(test_preds, test_y)
    return roc_auc_score(stud_y, stud_y), roc_auc_score(test_preds, test_y), roc_auc_score(val_preds, val_y)


model = LGBMClassifier(n_jobs=8
                           #, max_depth=4
                           # ,num_leaves=31
                           #, learning_rate=0.1
                           , n_estimators=100
                           # , max_bin=15
                           , colsample_bytree=0.8
                           , is_unbalance=True
                           , subsample=0.8
                           #,min_child_weight=6
                           )

regModel = LGBMRegressor(n_jobs=8
                           #, max_depth=12
                           #,num_leaves=256
                           , learning_rate=0.05
                           , n_estimators=100
                           #, max_bin=15
                           , colsample_bytree=0.2
                           #, is_unbalance=True
                           , subsample=0.5
                           #,min_child_weight=6
                           )


# All numerical, Genetic, ED, GCS, PMH
genetic = 1
ED = 1
GCS = 1
PMH = 1
CT = 1
SES = 1
SHX = 1
defa = 1
bio = 1
impute = 1
encode = 1
norm = 1

use_feats = np.array([genetic, ED, GCS, PMH, CT, SES, SHX, defa, bio, impute, encode, norm])
set_names =np.array(['Genetic', 'ED', 'GCS', 'PMH', 'CT', 'Survey', 'Substance Use', 'Demographic',
                     'Biomarkers', 'Imputed', 'Encoded', 'Norm'])
#print "Using selected features:"
#print set_names[use_feats>0]

X, num_names = get_X(genetic, ED, GCS, PMH, CT, SES, SHX, defa, bio, impute, encode, norm)

label_names = df.select_dtypes(exclude=[np.number]).columns.tolist()

#print 'Feature Shape:', X.shape
#CT_Intracraniallesion_FIN
#GCSMildModSevereRecode
#GOSE_OverallScore3M
#PTSD_6mo

#predict_one(model, 'CT_Intracraniallesion_FIN')
#X['CT_Intracraniallesion_FIN'] = df['CT_Intracraniallesion_FIN']
#X.to_csv('data/lesion_data.csv', index=False)
#loo(model, 'CT_Intracraniallesion_FIN')

reg_one(regModel, 'GOSE_OverallScore3M')
#reg_one(regModel, 'NeuroOverallRating3mo')
#top_cluster('GCSMildModSevereRecode')
#cluster(X, df, 'figs/tsne_admgcs_bio_snp', 'tsne')

#predict_one(model, 'CT_Intracraniallesion_FIN')

#predict_one(model, 'GOSE_OverallScore3M')

'''for x in range(2,20):
    tr, te, tv = private(model, 'CT_Intracraniallesion_FIN', x)
    print tr, te, tv'''

#plot(X, num_names)
#loo(model, 'HospDisch')
#predict_all(model)
#all_feat(model)
#predict_missing(regModel, X)