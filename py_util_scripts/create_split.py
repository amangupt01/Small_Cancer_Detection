import os
import sys
import os.path as osp
import numpy as np
import pickle


inner_dict = {"val":[], "P1":[], "P2":[], "P3":[], "P4":[], "P5":[]}
split_dict = {"mal":inner_dict.copy(), "ben":inner_dict.copy()}


#Do mal
img_lists = list(os.listdir('inbreast_data_final/test/mal/images/'))

np.random.shuffle(img_lists)
n = len(img_lists)
for i in range(n):
    if i <=1*0.8*0.2*n:
        split_dict["mal"]["P1"].append(img_lists[i])
    elif i <=2*0.8*0.2*n:
        split_dict["mal"]["P2"].append(img_lists[i])
    elif i <=3*0.8*0.2*n:
        split_dict["mal"]["P3"].append(img_lists[i])
    elif i <=4*0.8*0.2*n:
        split_dict["mal"]["P4"].append(img_lists[i])
    elif i <=5*0.8*0.2*n:
        split_dict["mal"]["P5"].append(img_lists[i])
    else:
        split_dict["mal"]["val"].append(img_lists[i])


img_lists = list(os.listdir('inbreast_data_final/test/ben/images/'))

np.random.shuffle(img_lists)
n = len(img_lists)
for i in range(n):
    if i <=1*0.8*0.2*n:
        split_dict["ben"]["P1"].append(img_lists[i])
    elif i <=2*0.8*0.2*n:
        split_dict["ben"]["P2"].append(img_lists[i])
    elif i <=3*0.8*0.2*n:
        split_dict["ben"]["P3"].append(img_lists[i])
    elif i <=4*0.8*0.2*n:
        split_dict["ben"]["P4"].append(img_lists[i])
    elif i <=5*0.8*0.2*n:
        split_dict["ben"]["P5"].append(img_lists[i])
    else:
        split_dict["ben"]["val"].append(img_lists[i])

pickle.dump(split_dict, open("Splits_dump.pickle",'wb'))

