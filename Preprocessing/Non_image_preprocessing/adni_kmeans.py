import pandas as pd
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='adni_kmeans')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--thres',type=str, default='0.9,0.9,0.9,0.9')
    return parser.parse_known_args()
args, unknown = parse_args()
K = args.K
clinical = pd.read_csv('./../non-image/ADNI/additional_adni1_2.csv')

clinical.drop(columns = ['ICV_bl','WholeBrain_bl','IMAGEUID_bl','Ventricles_bl','Fusiform_bl','MidTemp_bl','Month', 'M','race','education','Month_bl','Years_bl','ethnicity','apoe4','age_'],inplace=True)
total_index = []
for id in clinical['RID'].unique():
    local_index = []
    nan_count_index = []
    for idx in clinical[clinical['RID'] == id].index:
        local_index.append(idx)
        nan_count_index.append(clinical[clinical['RID'] == id].loc[idx].isna().sum())
    tmp = min(nan_count_index)
    total_index.append(local_index[nan_count_index.index(tmp)])
use_clinical = clinical.loc[total_index]

path_ = './../SimCLR/extracted_feature/adni/'
train_feature = np.loadtxt('./' + path_ + 'train_feature.csv',delimiter=',',dtype=np.float32)
valid_feature = np.loadtxt('./' + path_ + 'valid_feature.csv',delimiter=',',dtype=np.float32)
test_feature = np.loadtxt('./' + path_ + 'test_feature.csv',delimiter=',',dtype=np.float32)

train_id = pd.read_csv('./' + path_ +'train_id.csv', header=None)
valid_id = pd.read_csv('./' + path_ +'valid_id.csv', header=None)
test_id = pd.read_csv('./' + path_ +'test_id.csv', header=None)

id_list = []
image_list = []
for i in range(len(train_id)):
    a = str(train_id[0][i])
    l = int(a[:a[:-3].rfind('0000')])
    id_list.append(l)
    image_list.append(train_feature[i])
for i in range(len(valid_id)):
    a = str(valid_id[0][i])
    l = int(a[:a[:-3].rfind('0000')])
    id_list.append(l)
    image_list.append(valid_feature[i])
for i in range(len(test_id)):
    a = str(test_id[0][i])
    l = int(a[:a[:-3].rfind('0000')])
    id_list.append(l)
    image_list.append(test_feature[i])

use_col = use_clinical.keys().tolist()

for col in use_col:
    use_clinical[col].fillna(use_clinical[col].mode()[0],inplace=True)

k_means_list = use_col[3:]

from sklearn.preprocessing import minmax_scale
use_clinical_minmax = minmax_scale(use_clinical[k_means_list], axis=0, copy =True)
use_k_means = use_clinical.copy()
use_k_means[k_means_list] = use_clinical_minmax

transpose_num = use_k_means[k_means_list].T
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=K)
kmeans.fit(transpose_num)

type_list = []
for k in range(K):
    type_list.append([])
for i in range(len(k_means_list)):
    type_list[kmeans.labels_[i]].append(k_means_list[i])

non_image_feat = []
labels = []
use_clinical_dum = pd.get_dummies(use_clinical.drop(columns=['label','RID']),columns=use_col[3:7])
for id_ in id_list:
    lp = use_clinical[use_clinical['RID'] == id_].index.item()
    lab = use_clinical['label'].loc[lp]
    non_image_feat.append(minmax_scale(use_clinical_dum, axis=0, copy =True)[lp])
    # non_image_feat.append(use_clinical_dum.loc[lp])
    labels.append(lab)

import sklearn.metrics.pairwise
from sklearn.preprocessing import minmax_scale

k = 0
save_list= []
threses = args.thres.split(',')
threses = [float(th) for th in threses]
for types in type_list:
    before_adj_ = []
    use_clinical_dummy_multi = use_clinical[['label']+types]
    ll = use_clinical_dummy_multi.drop(columns=['label'])
    ll = ll.fillna(0)
    for id_ in id_list:
        lp = use_clinical[use_clinical['RID'] ==id_].index.item()
        p = ll.loc[lp].tolist()
        before_adj_.append(p)
    p_ = np.array(before_adj_)
    p_ = minmax_scale(p_, axis=0, copy =True)

    cos_ = sklearn.metrics.pairwise.cosine_similarity(p_,p_)
    adj_ = np.zeros(cos_.shape)
    thres = threses[k]
    for i in range(cos_.shape[0]):
        for j in range(cos_.shape[0]):
            if cos_[i][j] > thres:
                adj_[i][j] = 1
            else:
                adj_[i][j] = 0
    save_list.append(adj_)
    k+=1

y = np.zeros((len(labels),3))
for i in range(len(labels)):
    if labels[i] ==0:
        y[i,0]=1
    elif labels[i]==3:
        y[i,1]=1
    elif labels[i] ==4:
        y[i,2] = 1

concat_feature = []
for i in range(len(image_list)):
    concat = np.concatenate((np.expand_dims(image_list[i],axis=0),np.expand_dims(non_image_feat[i],axis=0)),axis=1)
    concat_feature.append(concat[0])

concate_feature_num = np.array(concat_feature)

indexes = [i for i in range(len(labels))]
train_ = len(indexes[:int(len(indexes)*0.6)])
valid_ = len(indexes[int(len(indexes)*0.6):int(len(indexes)*0.7)])
test_ = len(indexes[int(len(indexes)*0.7):])


import random
random.shuffle(indexes)
train_index = indexes[:train_]
valid_index = indexes[train_:train_+valid_]
test_index = indexes[train_+valid_:]

train_index = np.array(train_index)
valid_index = np.array(valid_index)
test_index = np.array(test_index)

if K ==6:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    adj_3 = save_list[3]
    adj_4 = save_list[4]
    adj_5 = save_list[5]
    multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'type5':adj_5,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==5:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    adj_3 = save_list[3]
    adj_4 = save_list[4]
    multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==4:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    adj_3 = save_list[3]
    multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}

elif K ==3:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==2:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    multi = {'label':y,'type0':adj_0,'type1':adj_1,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}

with open('./../MultiplexNetwork/data/adni.pkl', 'wb') as f:
    pickle.dump(multi, f, pickle.HIGHEST_PROTOCOL)