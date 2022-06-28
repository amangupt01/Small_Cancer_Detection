import os
from tqdm import tqdm
# in_path = "IRCHVal/4k/mal/gt/"
# out_path = "IRCHVal/1k/mal/gt/"

in_path = "Inbreast/4k/test/mal/gt/"
out_path = "Inbreast/1k/test/mal/gt/"

for filename in tqdm(os.listdir(in_path)):
    f = open(in_path+filename,"r")
    lines = f.readlines()
    f.close()
    out = open(out_path+filename,"w")
    for line in lines:
        print(line.split(" ")[1:])
        l = [float(i)/4 for i in line.split(" ")[1:]]
        out.write(" ".join([str(i) for i in l])+"\n")
    out.close()


