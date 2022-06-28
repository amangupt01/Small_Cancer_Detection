import cv2
import os
import shutil
from os.path import join
import numpy as np
from tqdm import tqdm


type = 'ben'
read_path_1k = f"ddsm_1k/{type}/"
read_path_2k = f"ddsm_2k/{type}/"

write_path_1k = f"ddsm_w_1k/" 
write_path_2k = f"ddsm_w_2k/" 

val_frac = 0.1
test_frac = 0.2
train_frac = 1 - val_frac - test_frac

gt_paths_1k = join(read_path_1k, 'gt')
gt_paths_2k = join(read_path_2k, 'gt')
img_paths_1k = join(read_path_1k, 'images')
img_paths_2k = join(read_path_2k, 'images')

num_img = len(os.listdir(img_paths_1k))
index = np.arange(num_img)

index = np.random.permutation(index)

img_names = []
for img_name in os.listdir(img_paths_1k):
    img_names.append(img_name[:-4])

for idx in tqdm(index):
    gt_name = img_names[idx] + ".txt"
    img_name = img_names[idx] + ".png"
    
    if idx < train_frac * num_img:
        curr_path_1k = join(write_path_1k, 'train')
        curr_path_2k = join(write_path_2k, 'train')

        shutil.copy(join(read_path_1k, 'images', img_name), join(curr_path_1k, 'images', img_name))
        shutil.copy(join(read_path_2k, 'images', img_name), join(curr_path_2k, 'images', img_name))
        
        # shutil.copy(join(read_path_1k, 'labels', gt_name), join(curr_path_1k, 'labels', gt_name))
        # shutil.copy(join(read_path_2k, 'labels', gt_name), join(curr_path_2k, 'labels', gt_name))

    elif idx < (train_frac + val_frac) * num_img:
        curr_path_1k = join(write_path_1k, 'val')
        curr_path_2k = join(write_path_2k, 'val')

        shutil.copy(join(read_path_1k, 'images', img_name), join(curr_path_1k, 'images', img_name))
        shutil.copy(join(read_path_2k, 'images', img_name), join(curr_path_2k, 'images', img_name))
        
        # shutil.copy(join(read_path_1k, 'labels', gt_name), join(curr_path_1k, 'labels', gt_name))
        # shutil.copy(join(read_path_2k, 'labels', gt_name), join(curr_path_2k, 'labels', gt_name))
    else:
        curr_path_1k = join(write_path_1k, 'test', type)
        curr_path_2k = join(write_path_2k, 'test', type)

        shutil.copy(join(read_path_1k, 'images', img_name), join(curr_path_1k, 'images', img_name))
        shutil.copy(join(read_path_2k, 'images', img_name), join(curr_path_2k, 'images', img_name))
        
        # shutil.copy(join(read_path_1k, 'gt', gt_name), join(curr_path_1k, 'gt', gt_name))
        # shutil.copy(join(read_path_2k, 'gt', gt_name), join(curr_path_2k, 'gt', gt_name))








