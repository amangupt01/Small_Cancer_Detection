import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import sys

root = 'DDSM_Full_Res'
# scale = int(sys.argv[1])
# part = sys.argv[2]

for scale in [2,4]:
    for part in ['Calc', 'Mass']:
        if scale == 2:
            scale_dir = '2k' 
        elif scale == 4:
            scale_dir = '4k'

        in_path_img = osp.join(root, scale_dir, part, "train", "images")
        in_path_labels = osp.join(root, scale_dir, part, "train", "labels")
        out_path_img = osp.join(root, f"Systematic_Crops_Of_{scale_dir}",  part, "images")
        out_path_labels = osp.join(root, f"Systematic_Crops_Of_{scale_dir}", part, "labels")


        dict_mals = {}
        offset = 0
        for i in os.listdir(in_path_labels):
            if i[-3:] == "txt":
                img_name = i[:-3] + "png"
                h,w,_ = cv2.imread(osp.join(in_path_img,img_name)).shape
                key_ = i[:-4]
                dict_mals[key_] = []
                for line in open(osp.join(in_path_labels,i)).readlines():
                    l = [float(j) for j in line.split(" ")]
                    n = [0 for _ in range(5-offset)]
                    n[1-offset] = int((l[1-offset]-l[3-offset]/2)*w)
                    n[2-offset] = int((l[2-offset]-l[4-offset]/2)*h)
                    n[3-offset] = int((l[1-offset]+l[3-offset]/2)*w)
                    n[4-offset] = int((l[2-offset]+l[4-offset]/2)*h)
                    dict_mals[key_].append(n[1:])

        # print(dict_mals)
        dirs = os.listdir(in_path_img)
        for img_name in tqdm(dirs):
            # img_name = "M_left_RAD100340_20150320_6_FILE110320876 (2)_MLO.png"
            if img_name[-3:] != "png":
                continue
            img_path = osp.join(in_path_img, img_name)
            img = cv2.imread(img_path)
            H, W, _  = img.shape

            ## (x1, y1, x2, y2)
            crop_coords = [] 
            for i in range(scale):
                for j in range(scale):
                    crop_coords.append( (i*W//scale, j*H//scale, (i+1)*W//scale, (j+1)*H//scale ))
            
            if img_name[:-4] in dict_mals:
                for crop_idx,(x1,y1,x2,y2) in enumerate(crop_coords):
                    label_path = osp.join(out_path_labels, img_name[:-4] +"_"+str(crop_idx)+ '.txt')
                    f = open(label_path,"w")
                    # print(dict_mals[img_name[:-4]])
                    for (mx1, my1, mx2, my2) in dict_mals[img_name[:-4]]:
                        # print(mx1,my1,mx2,my2)
                        if (mx2<=x1 or x2 <= mx1) or (my2<=y1 or y2 <= my1):
                            continue
                        nx1 = max(x1,mx1) - x1
                        nx2 = min(x2,mx2) - x1
                        ny1 = max(y1,my1) - y1
                        ny2 = min(y2,my2) - y1

                        to_write = [0 for _ in range(5)]
                        to_write[1] = ( nx1 + nx2 ) / (2 * (x2 - x1))
                        to_write[2] = ( ny1 + ny2 ) / (2 * (y2 - y1))
                        to_write[3] = ( nx2 - nx1 ) / (x2 - x1)
                        to_write[4] = ( ny2 - ny1 ) / (y2 - y1)
                        #print(to_write)
                        f.write(" ".join([str(_) for _ in to_write]) + "\n")
                        
                    f.close()     
                    img_crop = img[y1:y2,x1:x2]
                    cv2.imwrite(osp.join(out_path_img, img_name[:-4] +"_"+str(crop_idx)+ '.png'), img_crop)
            else:
                for crop_idx,(x1,y1,x2,y2) in enumerate(crop_coords):
                    img_crop = img[y1:y2,x1:x2]    
                    cv2.imwrite(osp.join(out_path_img, img_name[:-4] +"_"+str(crop_idx)+ '.png'), img_crop)
            # break



            
