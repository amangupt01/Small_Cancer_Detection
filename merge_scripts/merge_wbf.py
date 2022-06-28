import sys
import os
from os.path import join 
import torch
import shutil
from ensemble_boxes import *
import numpy as np
import cv2
from tqdm import tqdm

type = sys.argv[4]

img_1k_path = f"Small_Mass_Validation/1k/test/{type}/images/"
img_2k_path = f"Small_Mass_Validation/2k/test/{type}/images/"
img_4k_path = f"Small_Mass_Validation/4k/test/{type}/images/"

# img_1k_path = f"yolo_1k/AIIMS/test/{type}/images/"
# img_2k_path = f"yolo_2k/test/{type}/images/"
# img_4k_path = f"yolo_4k/test/{type}/images/"

# img_1k_path = f"DDSM/05k/all/test/images/"
# img_2k_path = f"DDSM/1k/all/test/images/"
# img_4k_path = f"DDSM/2k/all/test/images/"


# img_1k_path = f"IRCHVal/1k/{type}/images"
# img_2k_path = f"IRCHVal/2k/{type}/images"
# img_4k_path = f"IRCHVal/4k/{type}/images"

# img_1k_path = f"DDSM_Full_Res/1k/Mass/test/images/"
# img_2k_path = f"DDSM_Full_Res/2k/Mass/test/images/"
# img_4k_path = f"DDSM_Full_Res/4k/Mass/test/images/"

# img_1k_path = f"Inbreast/1k/test/{type}/images/"
# img_2k_path = f"Inbreast/2k/test/{type}/images/"
# img_4k_path = f"Inbreast/4k/test/{type}/images/"

exp_1k = sys.argv[1]
exp_2k = sys.argv[2]
exp_4k = sys.argv[3]

key_list = []
for img_name in os.listdir(img_1k_path):
    key_list.append(img_name[:-4])

for key_name in tqdm(key_list):
    curr_img_4k_path = join(img_4k_path,  f'{key_name}.png')
    img_4k = cv2.imread(curr_img_4k_path)
    
    h,w,c = img_4k.shape

    pred_4k = []
    conf_4k = []
    pred_path = f"runs/detect/exp{exp_4k}/labels/{key_name}.txt"
    if os.path.exists(pred_path):
        for line in open(pred_path).readlines():
            l = [float(i) for i in line.split(" ")[1:]]
            pred = l[:4]
            
            pred[0] /= w
            pred[1] /= h
            pred[2] /= w
            pred[3] /= h
            
            conf = l[4]
            
            pred_4k.append(pred)
            conf_4k.append(conf)
    else:
        pred_4k.append([0,0,0,0])
        conf_4k.append(0)


    
    pred_2k = []
    conf_2k = []
    pred_path = f"runs/detect/exp{exp_2k}/labels/{key_name}.txt"
    if os.path.exists(pred_path):
        for line in open(pred_path).readlines():
            l = [float(i) for i in line.split(" ")[1:]]
            pred = l[:4]
            
            pred[0] /= (w // 2)
            pred[1] /= (h // 2)
            pred[2] /= (w // 2)
            pred[3] /= (h // 2)
            
            conf = l[4]
            
            pred_2k.append(pred)
            conf_2k.append(conf)
    else:
        pred_2k.append([0,0,0,0])
        conf_2k.append(0)


    pred_1k = []
    conf_1k = []
    pred_path = f"runs/detect/exp{exp_1k}/labels/{key_name}.txt"
    if os.path.exists(pred_path):
        for line in open(pred_path).readlines():
            l = [float(i) for i in line.split(" ")[1:]]
            pred = l[:4]
            
            pred[0] /= (w // 4)
            pred[1] /= (h // 4)
            pred[2] /= (w // 4)
            pred[3] /= (h // 4)
            
            conf = l[4]
            
            pred_1k.append(pred)
            conf_1k.append(conf)
    else:
        pred_1k.append([0,0,0,0])
        conf_1k.append(0)

    
    old_boxes = [pred_1k, pred_2k, pred_4k]
    old_confs = [conf_1k, conf_2k, conf_4k]
    old_labels = [ [1]*len(conf_1k), [1]*len(conf_2k), [1]*len(conf_4k)]
    # print(boxes)
    # print(confs)
    # print(labels)
    weights=None
    iou_thr=0.55
    skip_box_thr=0.00
    conf_type='absent_model_aware_avg'
    allows_overflow=False

    boxes, scores, labels = weighted_boxes_fusion(old_boxes, old_confs, old_labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    # print(boxes)
    # print(scores)
    # print(labels)
    out_path = f"runs/combine_wbf/{exp_1k}-{exp_2k}-{exp_4k}-{type}/labels"
    os.makedirs(out_path, exist_ok=True)
    out = open(join(out_path,f'{key_name}.txt'),"w")
    for idx in range(len(boxes)):
        line = [0,0,0,0,0,0]
        line[5] = scores[idx]
        line[1] = boxes[idx][0] * ( w / 4)
        line[2] = boxes[idx][1] * ( h / 4)
        line[3] = boxes[idx][2] * ( w / 4)
        line[4] = boxes[idx][3] * ( h / 4)
        out.write(" ".join([str(_) for _ in line])+"\n")
    out.close()