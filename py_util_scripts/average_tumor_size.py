import os
import cv2
path  = "Small_Mass_Validation/1k/test/mal/"
avg_width = 0
avg_height = 0 
count = 0
num_img = 0
# for file_names in os.listdir(path+"labels/"):
#     img_name = file_names[:-3] + "png"
#     num_img += 1
#     img = cv2.imread(path+"images/"+img_name)
#     f = open(path+"labels/"+file_names,"r").readlines()
#     h,w,c = img.shape
#     for line in f:
#         count += 1
#         n = [float(i) for i in line.split(" ")[1:]]
#         avg_width += int(n[2]*w)
#         avg_height += int(n[3]*h)
w = []
h = []
for file_names in os.listdir(path+"gt/"):
    num_img +=1 
    f = open(path+"gt/"+file_names,"r").readlines()
    st = 0
    for line in f:
        count += 1
        n = [float(i) for i in line.split(" ")[st:]]
        avg_width += int(n[2])
        avg_height += int(n[3])
        w.append(int(n[2]))
        h.append(int(n[3]))

print("Number of Images = ", num_img)
print("Number of Cancers = ", count)
print("Min width", min(w))
print("Min height", min(h))
print("Average width = ", avg_width/count)
print("Average height = ", avg_height/count )
print("max width", max(w))
print("max height", max(h))
        


