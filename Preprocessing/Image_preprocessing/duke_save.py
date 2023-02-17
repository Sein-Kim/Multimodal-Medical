import SimpleITK as sitk
import numpy as np
import time
import matplotlib.pyplot as plt


data_path_txt = './../../medical_dataset/QIN/all_paths'
data_path = './../../medical_dataset/QIN/image_2D/'
f = open(data_path_txt,'r')
lines = f.readlines() 

i = 0
for line in lines:
    time_1 = time.time()
    ids_ = line.split('\t')[0]
    filename = line.split('\t')[1][:-1]
    
    image_2D_path = data_path + ids_ + '_'+str(i) + '.png'
    images = sitk.ReadImage(filename)
    images_array = sitk.GetArrayFromImage(images).astype('float32')
    img = np.squeeze(images_array)
    copy_img = img.copy()
    min = np.min(copy_img)
    max = np.max(copy_img)

    copy_img1 = copy_img - np.min(copy_img)
    copy_img = copy_img1/np.max(copy_img1)
    copy_img *= 2**8-1
    copy_img = copy_img.astype(np.uint8)
    plt.imshow(copy_img, cmap='gray')
    plt.axis('off')
    plt.savefig(image_2D_path, bbox_inches='tight',pad_inches = 0)
    plt.clf()
    
    del copy_img
    del copy_img1
    del images
    del images_array
    del img

    

    i+=1
