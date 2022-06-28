import os
import sys

detected = {}
confs = {
  1 : 0.232,
  2 : 0.146,
  4 : 0.161,
}
for scale in [1,2,4]:

  exp = {
    1 : f'runs/detect/exp{sys.argv[1]}',
    2 : f'runs/detect/exp{sys.argv[2]}',
    4 : f'runs/detect/exp{sys.argv[3]}',
  }


  gt_dir = f'Small_Mass_Validation/{scale}k/test/mal/gt'

  for gt_name in os.listdir(gt_dir):
    gt_path = f'{gt_dir}/{gt_name}'

    f = open(gt_path)
    lines = f.readlines()
    f.close()

    gx,gy,gw,gh = [float(_) for _ in lines[0].split()]

    name = gt_name[:-4]
    pred_path = f'{exp[scale]}/labels/{name}.txt'

    f = open(pred_path)
    lines = f.readlines()
    f.close()

    for line in lines:

      x1,y1,x2,y2, conf = [float(_) for _ in line.split()[1:] ]
      if conf < confs[scale]:
        continue

      xc = (x1+x2) / 2 
      yc = (y1+y2) / 2 

      if gx <= xc <= gx+gw and gy <= yc <= gy+gh:
        if name not in detected:
          detected[name] = set()
        detected[name].add(scale)

# print(detected)
for im in detected.keys():
  d = detected[im] 
  if d == {4}:
    print(im, d)




