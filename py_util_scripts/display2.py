import cv2
import os
from tqdm import tqdm
path = "DDSM/1k/test/mal/images/"
for img_name in tqdm(os.listdir(path)):
	img_name = img_name[:-4]
	img_path = "DDSM/1k/test/mal/images/"+img_name+".png"
	gt_path = "DDSM/1k/test/mal/gt/"+img_name+".txt"
	pred_path = "runs/detect/exp223/labels/"+img_name+".txt"

	img = cv2.imread(img_path)
	try:
		for line in open(pred_path,"r").readlines():
			n = [float(i) for i in line.split(" ")[1:]]
			if n[4] > 0.1:
				n = [int(i) for i in n]
				img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(0,0,255),4)
	except:
		pass
	for line in open(gt_path,"r").readlines():
		n = [int(float(i)) for i in line.split(" ")[1:5]]
		img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(0,255,0),4)
	cv2.imwrite("ddsm_viz/"+img_name+".png",img)                                          