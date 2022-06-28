import os
import cv2
from tqdm import tqdm
in_path = "/home/aman/scratch/yolov3/runs/detect/exp11/labels_old/"
out_path = "/home/aman/scratch/yolov3/runs/detect/exp11/labels/"

for filename in tqdm(os.listdir(in_path)):
    f = open(in_path+filename,"r")
    lines = f.readlines()
    f.close()
    img = cv2.imread(f"/home/aman/scratch/yolov5/yolo_1k/AIIMS/test/mal/images/{filename[:-4]}.png")
    h,w,c = img.shape
    out = open(out_path+filename,"w")
    for line in lines:
        l = [float(i) for i in line.split(" ")[1:]]
        x_c = (l[0])*w
        y_c = (l[1])*h
        bbox_w = l[2]*w
        bbox_h = l[3]*h 
        l[0] = x_c - bbox_w/2
        l[1] = y_c - bbox_h/2
        l[2] = x_c + bbox_w/2
        l[3] = y_c + bbox_h/2       
        out.write("0 "+" ".join([str(i) for i in l])+"\n")
    out.close()








# f = open("SmallMass/data/train/mal/train.txt","r").readlines()
# i = 0
# while(i<len(f)):
#     name = f[i][:-1]
#     num = int(f[i+1])
#     if num >= 0:
#         img = cv2.imread("SmallMass/data/train/mal/images_1k/"+name)
#         h,w,c = img.shape
#         out = open("SmallMass/data/train/labels/"+name.replace("png","txt"),"w")
#         for j in range(num):
#             l = [float(k) for k in f[i+j+2].split(" ")]
#             x_c = l[0] + l[2]/2
#             y_c = l[1] + l[3]/2
#             l[0] = x_c/w
#             l[1] = y_c/h
#             l[2] = l[2]/w
#             l[3] = l[3]/h
#             out.write("0 "+" ".join([str(k) for k in l[:4]])+"\n")
#         out.close()
#     i = i+num+2
#     if num == 0:
#         i += 1