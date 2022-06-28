import cv2
import sys
import numpy as np
import os
import os.path as osp
import random
from tqdm import tqdm

for scale in [2,4]:
  labels = f'Small_Mass_Validation/{scale}k/test/mal/gt'
  
  img_dir = f'Small_Mass_Validation/{scale}k/test/mal/images'
  out_path = f'Small_Mass_Validation/crops_ablation/{scale}k/mal'

  sum_w = sum_h = count = 0
  for label_name in os.listdir(labels):
    name = label_name[:-4]
    img_name = f'{img_dir}/{name}.png'
    # print(img_name)
    img = cv2.imread(img_name)
    H, W, _ = img.shape
    f = open(f'{labels}/{label_name}')

    for idx, line in enumerate(f.readlines()):
      gtx1,gty1,w,h = [float(j) for j in line.strip().split(' ')]
      xc = gtx1 + w // 2
      yc = gty1 + h // 2

      dim_w = int(w * 0.7)
      dim_h = int(h * 0.7)


      sum_h = max(sum_h, h)
      sum_w = max(sum_w, w)

      x1 = int(max(xc-dim_w, 0))
      x2 = int(min(xc+dim_w, W))
      y1 = int(max(yc-dim_h, 0))
      y2 = int(min(yc+dim_h, H))
      print(x1,x2,y1,y2)
      crop = img[y1:y2, x1:x2, :]

      cv2.imwrite(f'{out_path}/images/{name}_{idx}.png', crop)

      nx1 = gtx1 - x1
      ny1 = gty1 - y1
      print(nx1, ny1)
      assert(nx1 >= 0 and ny1 >= 0)

      lf = open(f'{out_path}/gt/{name}_{idx}.txt', 'w')
      lf.write(" ".join([str(_) for _ in [nx1,ny1,w,h]]) + '\n')

      lf.close()
    f.close()
  print(sum_h, sum_w)


