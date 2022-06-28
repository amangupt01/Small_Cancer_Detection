import os
import shutil
path = "/home/aman/scratch/SmallMass/data/train/"
for file_name in os.listdir(path+"ben/images_1k/"):
    shutil.copy(path+"ben/images_1k/"+file_name, path+"images/")
for file_name in os.listdir(path+"mal/images_1k/"):
    shutil.copy(path+"mal/images_1k/"+file_name, path+"images/")