from re import S
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
import random
data_path_txt = './../../medical_dataset/ADNI/all_paths.txt'
data_path = './../../medical_dataset/ADNI/image_2D/'

f = open(data_path_txt,'r')
lines = f.readlines()
all_data = []
for line in lines:
    name, path_ = line[:-1].split('\t')
    all_data.append([int(name),path_])
f.close()

labels = []
for data in all_data:
    labels.append(data[0])
    
i = 0
atlas = sitk.ReadImage('C:/Users/user/Desktop/ADNI1_Complete 1Yr 3T/ADNI/002_S_0413/MPR____N3__Scaled/2006-05-19_16_17_47.0/I40657/ADNI_002_S_0413_MR_MPR____N3__Scaled_Br_20070216232854688_S14782_I40657.nii')
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(atlas)
thres=-1
dict_data_count = {}

for data in all_data:
    if i >thres:
        time_1 = time.time()
        read_sitk = sitk.ReadImage(data[1])
        elastixImageFilter.SetMovingImage(read_sitk)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
        elastixImageFilter.Execute()
        read_sitk = elastixImageFilter.GetResultImage()

        img_vol = sitk.GetArrayFromImage(read_sitk)
        if img_vol.shape[0] >=100:
            side = [i+90 for i in range(40)]
            front = [i+100 for i in range(50)]
            cross = [i+90 for i in range(40)]
            random.shuffle(side)
            random.shuffle(front)
            random.shuffle(cross)
            image_2D_path = data_path +str(int(data[0])) +'_' +str(i) +'_'
            if not (data[0] in list(dict_data_count.keys())):
                dict_data_count[data[0]] = 1
                for s in side[:20]:
                    save_path = image_2D_path+ 'side' + str(s) +'.png'
                    plt.imshow(img_vol[:,:,s], cmap='gray')
                    plt.axis('off')
                    # plt.show
                    plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
                    del save_path
                plt.close('all')
                plt.clf()
                ##################
                for f in front[:20]:
                    save_path = image_2D_path+ 'front' + str(f) +'.png'
                    plt.imshow(img_vol[:,f], cmap='gray')
                    plt.axis('off')
                    # plt.show
                    plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
                    del save_path
                plt.close('all')
                plt.clf()
                ############################
                for c in cross[:20]:
                    save_path = image_2D_path+ 'cross' + str(c) +'.png'
                    plt.imshow(img_vol[50], cmap='gray')
                    plt.axis('off')
                    # plt.show
                    plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
                    del save_path
                plt.close('all')
                plt.clf()
                
                del read_sitk
                del img_vol
                del image_2D_path
    i+=1