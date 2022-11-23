import pandas as pd
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='qin_kmeans')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--thres',type=str, default='0.9,0.9,0.9,0.9')
    return parser.parse_known_args()
args, unknown = parse_args()
K = args.K
train_rate = 0.6
clinical = pd.read_csv('./../../non-image/OASIS3/clinical data 2.csv')
clinical_basic = pd.read_csv('./../../non-image/OASIS3/subject_basic.csv')
clinical_judge = pd.read_csv('./../../non-image/OASIS3/Judgement.csv')
clinical_basic_ = clinical_basic.drop(columns=['PETs','MR Sessions'], axis=1)
faq_list = ['BILLS','TAXES','SHOPPING','GAMES','STOVE','MEALPREP','EVENTS','PAYATTN','REMDATES','TRAVEL']
clinical_basic_[faq_list] = 0
clinical_basic_.drop(columns=['YOB'],inplace = True)
for col in clinical_basic_.columns:
    clinical_basic_[col] = clinical_basic_[col].fillna(clinical_basic_[col].mode()[0])
total_index = []
sign = 0
locate_cdr2=0
for id in clinical['Subject'].unique():
    local_index = []
    nan_count_index = []
    sign=0
    locate_cdr2 = 0
    for idx in clinical[clinical['Subject'] == id].index:
        local_index.append(idx)
        nan_count_index.append(clinical[clinical['Subject'] == id].loc[idx].isna().sum())
        if (clinical[clinical['Subject']==id].loc[idx]['cdr'] == 2.0):
            sign=1
            locate_cdr2 = idx
    tmp = min(nan_count_index)
    # total_index.append(local_index[nan_count_index.index(tmp)])

    if sign == 1:
        total_index.append(locate_cdr2)
    else:
        total_index.append(local_index[nan_count_index.index(tmp)])

use_clinical = clinical.loc[total_index]
use_clinical['label'] = 0
use_clinical['label'] = use_clinical['dx1'].apply(lambda x: 0 if (x=='Cognitively normal') else (2 if (x=='AD Dementia') else 1))
use_clinical['age_'] = pd.qcut(use_clinical['ageAtEntry'], 5, labels=False)
use_clinical['height_'] = pd.qcut(use_clinical['height'], 3, labels=False)
use_clinical['weight_'] = pd.qcut(use_clinical['weight'], 3, labels=False)

jud = clinical_judge[['Subject','DECSUB','DECIN','DECCLIN']]
for col in jud.columns:
    jud[col] = jud[col].fillna(jud[col].mode()[0])
li = ['DECSUB', 'DECIN', 'DECCLIN']
for il in li:
    clinical_basic_[il] = 0
    for sub in clinical_basic_['Subject']:
        try:
            a = jud[jud['Subject']==sub][il].max()
            clinical_basic_.loc[clinical_basic_[clinical_basic_['Subject']==sub].index,il] = a
        except:
            clinical_basic_.loc[clinical_basic_[clinical_basic_['Subject']==sub].index,il] = 0
            
clinical_basic_['cdr'] = use_clinical['cdr'].to_list()
use_clinical_2 = use_clinical[['Subject','cdr','age_','homehobb','apoe']]
use_clinical_2
for col in use_clinical_2.columns.tolist()[2:]:
    clinical_basic_[col] = use_clinical_2[col].to_list()
for col in clinical_basic_.columns:
    clinical_basic_[col] = clinical_basic_[col].fillna(clinical_basic_[col].mode()[0])
use_col = clinical_basic_.columns.tolist()[3:17] + clinical_basic_.columns.tolist()[-6:]
all_col = use_col+['Subject','cdr'] 

clinical_basic_all_dum = pd.get_dummies(clinical_basic_[use_col], columns=use_col)
sub = clinical_basic_['Subject']
cdr = clinical_basic_['cdr']
# clinical_basic_all_dum.drop(columns=['Subject','cdr'], inplace=True)
feature_dict = {}
label_dict = {}
i = 0

for s in sub:
    feature_dict[s] = clinical_basic_all_dum.iloc[i].to_numpy()
    label_dict[s] = cdr[i]
    i+=1
non_img = {'label': label_dict, 'feature': feature_dict}
with open('./../non_image/OASIS3/oasis_nonimg.pkl', 'wb') as f:
    pickle.dump(non_img, f, pickle.HIGHEST_PROTOCOL)
path_ = '../SimCLR/extracted_feature/oasis/'
train_feature = np.loadtxt('./' + path_ + 'train_feature.csv',delimiter=',',dtype=np.float32)
# valid_feature = np.loadtxt('./' + path_ + 'valid_feature.csv',delimiter=',',dtype=np.float32)
test_feature = np.loadtxt('./' + path_ + 'test_feature.csv',delimiter=',',dtype=np.float32)

train_id = pd.read_csv('./' + path_ +'train_id.csv', header=None)
# valid_id = pd.read_csv('./' + path_ +'valid_id.csv', header=None)
test_id = pd.read_csv('./' + path_ +'test_id.csv', header=None)


with open('./../non-image/OASIS3/oasis_nonimg.pkl', 'rb') as fr:
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
    a = 'OAS' + str(train_id[0][i])[:5]
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
    a = 'OAS' + str(test_id[0][i])[:5]
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
    if i == 0.0:
        modi_label.append(0)
    elif i == 0.5:
        modi_label.append(1)
    elif i ==1.0:
        modi_label.append(2)
    else:
        modi_label.append(3)

labels = modi_label
indexes = train_index + valid_index + test_index
# indexes=train_index.tolist() + valid_index.tolist() + test_index.tolist()
train_ = len(indexes[:int(len(indexes)*train_rate)])
valid_ = len(indexes[int(len(indexes)*train_rate):int(len(indexes)*(train_rate+0.1))])
test_ = len(indexes[int(len(indexes)*(train_rate+0.1)):])

import random
random.shuffle(indexes)
train_index = indexes[:train_]
valid_index = indexes[train_:train_+valid_]
test_index = indexes[train_+valid_:]
# train_index = [i for i in range(train_)]
# valid_index = [i+max(train_index)+1 for i in range(valid_)]
# test_index = [i+max(valid_index)+1 for i in range(test_)]

before_adj_num = np.array(before_adj)
all_feature_num = np.array(all_feature)
concate_feature_num = np.array(concate_feature)
import sklearn.metrics.pairwise
cos_sim = sklearn.metrics.pairwise.cosine_similarity(before_adj_num,before_adj_num)

adj = np.zeros(cos_sim.shape)
thres = 0.4#0.925
for i in range(cos_sim.shape[0]):
    for j in range(cos_sim.shape[0]):
        if cos_sim[i][j] > thres:
            adj[i][j] = 1
        else:
            adj[i][j] = 0
print('dense')
print(sum(sum(adj))/(adj.shape[0]*adj.shape[0]))
# print(use_col)
# k_means_list = use_col[:-4] + use_col[-3:]
k_means_list = ['DECSUB', 'DECIN', 'DECCLIN', 'age_', 'homehobb','apoe','UDS B9: Clin. Judgements','UDS B5: NPI-Q','UDS B8: Phys. Neuro Findings','Psych Assessments']

# ['UDS B9: Clin. Judgements', 'UDS A5: Sub Health Hist.', 'UDS B6: GDS', 'UDS A1: Sub Demos']
# ['UDS B7: FAQs', 'UDS A2: Informant Demos', 'age_', 'UDS B5: NPI-Q']
# ['DECCLIN', 'UDS A3: Partcpt Family Hist.', 'UDS B3: UPDRS', 'ADRC Clinical Data']
# ['UDS B2: HIS and CVD', 'homehobb', 'UDS D1: Clinician Diagnosis', 'UDS B8: Phys. Neuro Findings', 'Psych Assessments', 'DECIN', 'apoe']

transpose_num = clinical_basic_[k_means_list].T
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters=K)
kmeans.fit(transpose_num)
type_list = []
for k in range(K):
    type_list.append([])
for i in range(len(k_means_list)):
    type_list[kmeans.labels_[i]].append(k_means_list[i])
feature_dict = {}
label_dict = {}
i = 0
from sklearn.preprocessing import minmax_scale

k = 0
save_list= []
threses = args.thres.split(',')
threses = [float(th) for th in threses]
for types in type_list:
    before_adj_ = []
    use_clinical_dummy_multi = pd.get_dummies(clinical_basic_[['cdr']+types], columns=types)
    ll = use_clinical_dummy_multi.drop(columns=['cdr'])
    ll = ll.fillna(0)
    for id_ in id_list:
        lp = clinical_basic_[clinical_basic_['Subject'] ==id_].index.item()
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

y = np.zeros((len(labels),4))
for i in range(len(labels)):
    if labels[i] ==0:
        y[i,0]=1
    elif labels[i]==1:
        y[i,1]=1
    elif labels[i] ==2:
        y[i,2] = 1
    else:
        y[i,3] =1

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
    oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'type5':adj_5,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==4:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    adj_3 = save_list[3]
    oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}

elif K ==5:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    adj_3 = save_list[3]
    adj_4 = save_list[4]
    oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==3:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    adj_2 = save_list[2]
    oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
elif K ==2:
    adj_0 = save_list[0]
    adj_1 = save_list[1]
    oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}

with open('./../MultiplexNetwork/data/oasis.pkl', 'wb') as f:
    pickle.dump(oasis3_multi, f, pickle.HIGHEST_PROTOCOL)