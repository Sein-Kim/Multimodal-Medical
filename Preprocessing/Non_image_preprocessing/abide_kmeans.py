import pandas as pd
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='abide_kmeans')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--thres',type=str, default='0.9,0.9,0.9,0.9')
    return parser.parse_known_args()
args, unknown = parse_args()
K = args.K
clinical = pd.read_csv('./../non-image/ABIDE/FinalMerged_MM.csv')

clinical.drop(columns=['SITE_ID'], inplace=True)
clinical.drop(columns = ['FIQ_TEST_TYPE', 'VIQ_TEST_TYPE', 'PIQ_TEST_TYPE'], inplace= True)
clinical['HANDEDNESS_CATEGORY']= clinical['HANDEDNESS_CATEGORY'].fillna(clinical['HANDEDNESS_CATEGORY'].mode()[0])
clinical = clinical.fillna(-9999)	

use_l = ['SUB_ID',
 'DX_GROUP',
 'AGE_AT_SCAN',
 'SEX',
 'HANDEDNESS_CATEGORY',
 'HANDEDNESS_SCORES',
 'FIQ',
 'VIQ',
 'PIQ',
 'ADOS_STEREO_BEHAV',
 'ADOS_GOTHAM_SOCAFFECT',
 'ADOS_GOTHAM_RRB',
 'ADOS_GOTHAM_TOTAL',
 'ADOS_GOTHAM_SEVERITY',
 'SRS_RAW_TOTAL',
 'EYE_STATUS_AT_SCAN']
clinical_dum = clinical[use_l]
df= clinical_dum.drop(columns = ['SUB_ID','DX_GROUP'])
normalized_df=((df-df.mean())/df.std()).fillna(0)
feature_dict = {}
label_dict = {}
i = 0
sub_id = clinical_dum['SUB_ID']
use_clinical_dummy_no = normalized_df

for sub in sub_id:
    feature_dict[sub] = use_clinical_dummy_no.iloc[i].to_numpy()
    label_dict[sub] = clinical_dum['DX_GROUP'].iloc[i]
    i+=1
non_img = {'label': label_dict, 'feature': feature_dict}
with open('./../non_image/ABIDE/abide_nonimg.pkl', 'wb') as f:
    pickle.dump(non_img, f, pickle.HIGHEST_PROTOCOL)

path_ = './../SimCLR/extracted_feature/abide/'
train_feature = np.loadtxt('./' + path_ + 'train_feature.csv',delimiter=',',dtype=np.float32)
# valid_feature = np.loadtxt('./' + path_ + 'valid_feature.csv',delimiter=',',dtype=np.float32)
test_feature = np.loadtxt('./' + path_ + 'test_feature.csv',delimiter=',',dtype=np.float32)

train_id = pd.read_csv('./' + path_ +'train_id.csv', header=None)
# valid_id = pd.read_csv('./' + path_ +'valid_id.csv', header=None)
test_id = pd.read_csv('./' + path_ +'test_id.csv', header=None)

with open('./../non-image/ABIDE/abide_nonimg.pkl', 'rb') as fr:
    data = pickle.load(fr)
before_adj = []
labels= []
train_index = []
valid_index = []
test_index = []


all_feature = []
concate_feature = []#brain feature + patient feature
k = 0
id_list = []
for i in range(len(train_id)):
    a = int(str(train_id[0][i])[:5])
    id_list.append(a)
    l = data['label'][a]
    l_ = data['feature'][a]
    # l = rid_label_dict[id_[i]]
    train_index.append(k)

    labels.append(l)
    before_adj.append(l_)
    all_feature.append(list(train_feature[k]))
    concate_feature.append(list(train_feature[k]) + list(l_))
    k+=1

k_test = 0
for i in range(len(test_id)):
    a = int(str(test_id[0][i])[:5])
    id_list.append(a)
    l = data['label'][a]
    l_ = data['feature'][a]
    # l = rid_label_dict[id_[i]]
    test_index.append(k)

    labels.append(l)
    before_adj.append(l_)
    all_feature.append(list(test_feature[k_test]))
    concate_feature.append(list(test_feature[k_test]) + list(l_))
    k+=1
    k_test +=1


modi_label = []
for i in labels:
    if i == 1:
        modi_label.append(0)
    else:
        modi_label.append(1)


labels = modi_label

indexes = train_index + valid_index + test_index
train_ = len(indexes[:int(len(indexes)*0.6)])
valid_ = len(indexes[int(len(indexes)*0.6):int(len(indexes)*0.7)])
test_ = len(indexes[int(len(indexes)*0.7):])

import random
random.shuffle(indexes)
train_index = indexes[:train_]
valid_index = indexes[train_:train_+valid_]
test_index = indexes[train_+valid_:]
before_adj_num = np.array(before_adj)
all_feature_num = np.array(all_feature)
concate_feature_num = np.array(concate_feature)
from sklearn.preprocessing import minmax_scale
before_adj_num = minmax_scale(before_adj_num, axis=0, copy =True)

import sklearn.metrics.pairwise
cos_sim = sklearn.metrics.pairwise.cosine_similarity(before_adj_num,before_adj_num)

will_remove_list_1 = []
will_remove_list_2 = []
for id_ in id_list:
    if  clinical[clinical['SUB_ID'] ==id_]['ADI_R_SOCIAL_TOTAL_A'].item() == -9999.0:
        will_remove_list_1.append(id_)
    if clinical[clinical['SUB_ID'] ==id_]['SRS_RAW_TOTAL'].item() == -9999.0:
        will_remove_list_2.append(id_)

transpose_num = clinical[use_l[2:]].T
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=K)
kmeans.fit(transpose_num)
type_list = []
for k in range(K):
    type_list.append([])
for i in range(len(use_l[2:])):
    type_list[kmeans.labels_[i]].append(use_l[2:][i])

from sklearn.preprocessing import minmax_scale

k = 0
threses = args.thres.split(',')
threses = [float(th) for th in threses]
save_list= []
for types in type_list:
    before_adj_ = []
    print('type' + str(k))
    print("********")
    ll = clinical[['DX_GROUP', 'SUB_ID']+types].drop(columns=['DX_GROUP'])
    ll = ll.fillna(0)
    for id_ in id_list:
        lp = clinical[clinical['SUB_ID'] ==id_].index.item()
        p = ll.drop(columns=['SUB_ID']).loc[lp].tolist()
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
                if k ==2:
                    if id_list[i] in will_remove_list_1:
                        adj_[i][j] = 0
                    if id_list[j] in will_remove_list_1:
                        adj_[i][j] = 0
                if k ==3:
                    if id_list[i] in will_remove_list_2:
                        adj_[i][j] = 0
                    if id_list[j] in will_remove_list_2:
                        adj_[i][j] = 0   
            else:
                adj_[i][j] = 0
    for i in range(cos_.shape[0]):
        for j in range(cos_.shape[0]):
            if i == j:
                adj_[i][j] = 1
    save_list.append(adj_)
    k+=1

y = np.zeros((len(labels),2))
for i in range(len(labels)):
    if labels[i] ==0:
        y[i,0]=1
    else:
        y[i,1]=1

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
  
with open('./../MultiplexNetwork/data/abide.pkl', 'wb') as f:
    pickle.dump(multi, f, pickle.HIGHEST_PROTOCOL)