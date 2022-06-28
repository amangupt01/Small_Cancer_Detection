import cv2
import os
#names = ['right_103485711_20180123_5_FILE3271_CC.png', 'left_103141839_20180116_6_FILE6462_MLO.png', 'left_103619019_20180319_5_FILE7725_MLO.png', 'right_103146419_20180108_5_FILE668734498_MLO.png', 'left_103019332_20180109_6_FILE7334_CC.png', 'left_103499558_20180130_6_FILE830135082_CC.png', 'left_103489800_20180131_5_FILE0658_CC.png', 'right_103447736_20180201_6_FILE536213374_MLO.png', 'right_103590336_20180307_5_FILE096211826_CC.png', 'right_103449785_20180112_4c_FILE0979_MLO.png', 'left_103501930_20180201_6_FILE9552_MLO.png', 'left_103669742_20180411_5_FILE848935156_CC.png', 'right_102990465_20180205_5_FILE7140_MLO.png', 'left_103555540_20180222_6_FILE7046_MLO.png', 'left_103204627_20180110_5_FILE3096_MLO.png', 'left_103509366_20180207_4b_FILE5712_MLO.png']
names = ['right_103497037_20180201_4c_FILE7262__MLO.png', 'left_103204627_20180110_5_FILE3095_CC.png', 'right_103447987_20180110_5_FILE0331_CC.png', 'left_103636366_20180406_4c_FILE280712473_MLO.png', 'right_103630155_20180326_5_FILE729814064_CC.png', 'left_103425769_20180418_6_FILE6348_MLO.png', 'left_103628934_20180321_5_FILE8123_CC.png', 'left_103552454_20180306_5_FILE4516_CC.png', 'left_103653287_20180403_4b_FILE5422_CC.png', 'right_103485711_20180123_5_FILE3271_CC.png', 'left_103141839_20180116_6_FILE6462_MLO.png', 'left_103619019_20180319_5_FILE7725_MLO.png', 'right_103146419_20180108_5_FILE668734498_MLO.png']
names = ['26400000_82441186.png']
pred_1k_path = "/home/aman/scratch/yolov5/runs/detect/exp292/labels/"
pred_2k_path = "/home/aman/scratch/yolov5/runs/detect/exp96/labels/"
pred_4k_path = "/home/aman/scratch/yolov5/runs/detect/exp92/labels/"
nms = "/home/aman/scratch/yolov5/runs/combine/97-96-92/labels/"
for name in names :
    path = "Small_Mass_Validation/4k/test/mal/"
    img = cv2.imread(path+"images/"+name)

    f =open(pred_1k_path+name[:-4]+".txt","r").readlines()
    for line in f:
        l = line.split(" ")
        n = [int(float(i)) for i in l[1:]]
        if float(str(l[5])) > 0.002:
            img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(0,0,255),4)
            #img = cv2.putText(img, str(l[5]), (n[0], n[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # f =open(pred_2k_path+name[:-4]+".txt","r").readlines()
    # for line in f:
    #     l = line.split(" ")
    #     n = [int(float(i)/2) for i in l[1:]]
    #     if float(str(l[5])) > 0.1:
    #         img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(0,255,0),4)
    #         #img = cv2.putText(img, str(l[5]), (n[0], n[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    # f =open(pred_4k_path+name[:-4]+".txt","r").readlines()
    # for line in f:
    #     l = line.split(" ")
    #     n = [int(float(i)/4) for i in l[1:]]
    #     if float(str(l[5])) > 0.05:
    #         img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(255,0,0),4)
    #         #img = cv2.putText(img, str(l[5]), (n[0], n[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    # # count = 0 
    # f =open(nms+name[:-4]+".txt","r").readlines()
    # for line in f:
    #     l = line.split(" ")
    #     n = [int(float(i)) for i in l[1:]]
    #     if float(str(l[5])) > 0.1:
    #         img = cv2.rectangle(img,(n[0],n[1]), (n[2],n[3]),(255,255,255),4)

    cv2.imwrite("visualize/mal/"+name,img)
    # cv2.imwrite("dump2.png",img)