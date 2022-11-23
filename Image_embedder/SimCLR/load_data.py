import os
import numpy as np
import torch
import torchvision.transforms as transforms

from tqdm import tqdm
import cv2


def load_data(args):
    tf = transforms.ToTensor()
    data_name = args.data
    if data_name =='ADNI':
        root_dir = "./../../medical_dataset/ADNI/image_2D/"
    elif data_name =='OASIS':
        root_dir = "./../../medical_dataset/OASIS/image_2D/"
    elif data_name =='ABIDE':
        root_dir = "./../../medical_dataset/ABIDE/image_2D/"
    elif data_name =='CMMD':
        root_dir = "./../../medical_dataset/CMMD/image_2D/"
    elif data_name =='QIN':
        root_dir = './../../medical_dataset/QIN/image_2D/'

    files = os.listdir(root_dir)
    i=0
    name_list = []
    for file in tqdm(files):
        if i ==0:

            img = cv2.imread(root_dir +'/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_three = cv2.merge([gray,gray,gray])
            img__ = cv2.resize(gray_three, (96,96), interpolation = cv2.INTER_AREA)
            # with torch.cuda.device(args.gpu_index):
            # img_t = torch.tensor(img__, dtype=torch.float, device=args.device)
            with torch.cuda.device(args.gpu_index):
                img_t = tf(img__).cuda()
            # print(args.device)
            # print(img_t)
            images = img_t.unsqueeze(0)
            if 'CMMD' in root_dir:
                id_ = file.split('-')[0][-1] +file.split('-')[1][0:4]
                num = '0'
            else:
                id_, num, coordinate = file.split('_')
            #0000 for escape from unexpected duplication of id + num
            names = int(id_ +'0000' + num)
            name_list.append(names)

            del img
            del gray
            del gray_three
            del img__
            del img_t
        else:
            img = cv2.imread(root_dir +'/'+file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_three = cv2.merge([gray,gray,gray])
            img__ = cv2.resize(gray_three, (96,96), interpolation = cv2.INTER_AREA)
            with torch.cuda.device(args.gpu_index):
                img_t = tf(img__).cuda()
            images = torch.cat((images,img_t.unsqueeze(0)),0)

            if 'CMMD' in root_dir:
                id_ = file.split('-')[0][-1] +file.split('-')[1][0:4]
                num = '0'
            else:
                id_, num, coordinate = file.split('_')
            names = int(id_ +'0000' + num)
            name_list.append(names)
            
            del img
            del gray
            del gray_three
            del img__
            del img_t
        i+=1
    train_images = images.detach().cpu().numpy()
    train_names = np.expand_dims(np.array(name_list),axis=1)

    
    unique_label = list(set(name_list))
    labels_np = np.array(name_list)
    ii = 0
    image_list = []
    label_list__=[]
    for l in tqdm(unique_label):
        if ii==0:
            a = np.where(labels_np == l)
            index_ = list(a[0])
            coor_img_t = images[index_]
            image_list.append(coor_img_t)
            label_list__.append(l)
            del coor_img_t

        else:
            a = np.where(labels_np == l)
            index_ = list(a[0])
            coor_img_t = images[index_]
            image_list.append(coor_img_t)
            label_list__.append(l)
            del coor_img_t
        ii+=1
    del iamges
    return train_images,train_names, image_list, label_list__
