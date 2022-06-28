from itertools import count
import os
mal_pred_path = "runs/detect/exp153/labels/"
mal_gt_path = "Small_Mass_Validation/1k/test/mal/gt/"


def true_positive(gt, pred):
    # If center of pred is inside the gt, it is a true positive
    #print(pred)
    c_pred = ((pred[0]+pred[2])/2., (pred[1]+pred[3])/2.)
    if (c_pred[0] >= gt[0] and c_pred[0] <= gt[2] and
            c_pred[1] >= gt[1] and c_pred[1] <= gt[3]):
        return True
    return False

min_conf_tp = 1
avg = 0 
c = 0 
for pred_file in os.listdir(mal_pred_path):
    gts = open(mal_gt_path + pred_file,"r").readlines()
    preds = open(mal_pred_path + pred_file,"r").readlines()
    if len(gts)!= 1:
        print(len(gts))
    for gt in gts:
        print
        bbox_gt = [float(i) for i in gt.split(" ")]
        bbox_gt[2] += bbox_gt[0]
        bbox_gt[3] += bbox_gt[1] 
        list_conf_tp = []
        for pred in preds:
            bbox_pred = [float(i) for i in pred.split(" ")[1:]]
            #print(bbox_pred)
            conf = bbox_pred[4]
            bbox_pred = bbox_pred[:-1]
            if true_positive(bbox_gt,bbox_pred):
                list_conf_tp.append(conf)
        if list_conf_tp != []:
            c += 1
            avg += max(list_conf_tp)
            min_conf_tp = min(min_conf_tp,max(list_conf_tp))
            #print(max(list_conf_tp))
print("Minimum confidence for true positive is ", min_conf_tp)
print("Number of True Positives is ", c)
print("Average confidence for true positive is", avg/c)



