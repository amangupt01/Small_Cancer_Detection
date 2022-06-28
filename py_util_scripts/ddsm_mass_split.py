import shutil
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import cv2



def train_val():

  train_path = f'DDSM/2k/all/train/images'
  val_path = f'DDSM/2k/all/val/images'

  train_label_src = f'DDSM/2k/all/train/labels'
  val_label_dst = f'DDSM/2k/all/val/labels'

  images = list(os.listdir(train_path))
  num_images = len(images) 

  num_val = int(0.2 * num_images)
  indices = np.random.permutation(range(num_images))[:num_val]

  for idx in tqdm(indices):
    img_name = images[idx]

    name = img_name[:-4]
    label_name = name + '.txt'

    shutil.move(f'{train_path}/{img_name}', f'{val_path}/{img_name}')
    shutil.move(f'{train_label_src}/{label_name}', f'{val_label_dst}/{label_name}')
      

def copy_scales():
  scale_dirs = ['1k', '05k']
  scales = [2,4]

  for i in range(2):
    scale = scales[i]
    scale_dir = scale_dirs[i]

    for split in ['train', 'test', 'val']:
      img_src = f'DDSM/2k/all/{split}/images'
      img_dst = f'DDSM/{scale_dir}/all/{split}/images'

      label_pref = "gt" if split == "test" else "labels"

      label_src = f'DDSM/2k/all/{split}/{label_pref}'
      label_dst = f'DDSM/{scale_dir}/all/{split}/{label_pref}'

      for img_name in tqdm(os.listdir(img_src)):
        img = cv2.imread(f'{img_src}/{img_name}')

        new_img = cv2.resize(img, (0,0), fx=1/scale, fy=1/scale)
        cv2.imwrite(f'{img_dst}/{img_name}', new_img)

        label_name = img_name[:-4] + '.txt'

        if split == 'test':
          fin = open(f'{label_src}/{label_name}')
          fout = open(f'{label_dst}/{label_name}', 'w')

          for line in fin.readlines():
            vals = line.split()
            bbox = [str(float(elem) / scale) for elem in vals[1:]]
            new_vals = [vals[0]] + bbox

            fout.write(' '.join(new_vals) + '\n')
            
          fin.close()
          fout.close()
        
        else:
          shutil.copy(f'{label_src}/{label_name}', f'{label_dst}/{label_name}')

          
            


# test_train()
# train_val()
copy_scales()