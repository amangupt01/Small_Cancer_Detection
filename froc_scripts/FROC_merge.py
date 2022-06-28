import os
import glob
import sys
import numpy as np
from os.path import join


'''
    Note: Anywhere empty boxes means [] and not [[]]
'''


def remove_true_positives(gts, preds):
    if type(preds) == list:
        preds = np.array(preds)
    #print(type(preds),preds.shape)
    #print(preds)
    if len(preds) == 0:
        return preds, 0, len(gts)

    def true_positive(gt, pred):
        # If center of pred is inside the gt, it is a true positive
        #print(pred)
        c_pred = ((pred[0]+pred[2])/2., (pred[1]+pred[3])/2.)
        if (c_pred[0] >= gt[0] and c_pred[0] <= gt[2] and
                c_pred[1] >= gt[1] and c_pred[1] <= gt[3]):
            return True
        return False

    def true_positives(gt, preds):
        # If center of pred is inside the gt, it is a true positive
        #print(pred)
        c_preds0, c_preds1 = (preds[:,0]+preds[:,2])/2., (preds[:,1]+preds[:,3])/2.
        return (c_preds0 >= gt[0]) & (c_preds0 <= gt[2]) & (c_preds1 >= gt[1]) & (c_preds1 <= gt[3])

    tps = 0
    fns = 0

    for gt in gts:
        # First check if any true positive exists
        # If more than one exists, do not include it in next set of preds
        # add_tp = False
        # new_preds = []
        # for pred in preds:
        #     if true_positive(gt, pred):
        #         add_tp = True
        #     else:
        #         new_preds.append(pred)
        # preds = new_preds
        # if add_tp:
        #     tps += 1
        # else:
        #     fns += 1
        mask = true_positives(gt, preds)
        preds = preds[~mask]
        if True in mask:
            tps += 1
        else:
            fns += 1
    return preds, tps, fns



def calc_metric_single(gts, preds, threshold,):
    '''
        Returns fp, tp, tn, fn
    '''
    preds = list(filter(lambda x: x[0] >= threshold, preds))
    preds = [pred[1:] for pred in preds]  # Remove the scores

    if len(gts) == 0:
        return len(preds), 0, 1 if len(preds) == 0 else 0, 0
    preds, tps, fns = remove_true_positives(gts, preds)
    #print(threshold, tps, fns)
    # All remaining will have to fps
    fps = len(preds)
    return fps, tps, 0, fns


def calc_metrics_at_thresh(im_dict, threshold):
    '''
        Returns fp, tp, tn, fn
    '''
    fps, tps, tns, fns = 0, 0, 0, 0
    for key in im_dict:
        #print(key)
        fp,tp,tn,fn = calc_metric_single(im_dict[key]['gt'],
                           im_dict[key]['preds'], threshold)
        # if fn > 0 and threshold ==0.:
        #     print(key)
        fps+=fp
        tps+=tp
        tns+=tn
        fns+=fn

    return fps, tps, tns, fns


def calc_froc_from_dict(im_dict, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3], save_to = None):

    num_images = len(im_dict)

    gap = 0.001
    n = int(1/gap)
    thresholds = [i * gap for i in range(n)]
    fps = [0 for _ in range(n)]
    tps = [0 for _ in range(n)]
    tns = [0 for _ in range(n)]
    fns = [0 for _ in range(n)]

    for i,t in enumerate(thresholds):
        fps[i], tps[i], tns[i], fns[i] = calc_metrics_at_thresh(im_dict, t)



    vv = 30
    print(fps[vv], tps[vv], tns[vv], fns[vv])

    # Now calculate the sensitivities
    senses = []
    for t,f in zip(tps, fns):
        try: senses.append(t/(t+f))
        except: senses.append(0.)

    if save_to is not None:
        f = open(save_to, 'w')
        for fp,s in zip(fps, senses):
            f.write(f'{fp/num_images} {s}\n')
        f.close()

    senses_req = []
    for fp_req in fps_req:
        for i,f in enumerate(fps):
            if f/num_images < fp_req:
                if fp_req == 0.1:
                    pass
                    #print(fps[i], tps[i], tns[i], fns[i])
                senses_req.append(senses[i-1])
                break
    return senses_req, fps_req




def file_to_bbox(file_name):
    try:
        content = open(file_name, 'r').readlines()
        st = 0
        if len(content) == 0:
            # Empty File Should Return []
            return []
        if content[0].split()[0].isalpha():
            st = 1
        l = [[float(x) for x in line.split()[st:]] for line in content]
        return [[j[0],j[1],j[0]+j[2],j[1]+j[3]] for j in l]
    except FileNotFoundError:
        print(f'No Corresponding Box Found for file {file_name}, using [] as preds')
        return []
    except Exception as e:
        print('Some Error',e)
        return []


def file_to_bbox_yolo(file_name):
    try:
        content = open(file_name, 'r').readlines()
        st = 1
        if len(content) == 0:
            # Empty File Should Return []
            return []
        l = [[float(x) for x in line.split()[st:]] for line in content]
        return [[j[4],j[0],j[1],j[2],j[3]] for j in l]
    except FileNotFoundError:
        # print(f'2) No Corresponding Box Found for file {file_name}, using [] as preds')
        return []
    except Exception as e:
        print('Some Error',e)
        return []

def generate_image_dict(mal_exp, ben_exp,
                        root_fol='/home/krithika_1/densebreeast_datasets/AIIMS_C1',
                        mal_path=None, ben_path=None, gt_path=None,
                        mal_img_path = None, ben_img_path = None
                        ):

    mal_path = join(root_fol, mal_exp,"labels")
    ben_path = join(root_fol, ben_exp,"labels")
    yolo_model = "Small_Mass_Validation/1k" # TODO take input
    mal_img_path = yolo_model+"/test/mal/images"
    ben_img_path = yolo_model+"/test/ben/images"
    gt_path = yolo_model+"/test/mal/gt/"
    '''
        image_dict structure:
            'image_name(without txt/png)' : {'gt' : [[...]], 'preds' : [[]]}
    '''
    image_dict = dict()

    # GT Might be sightly different from images, therefore we will index gts based on
    # the images folder instead.
    for file in os.listdir(mal_img_path):
    # for file in glob.glob(join(gt_path, '*.txt')):
        if not file.endswith('.png'):
            continue
        file = file[:-4] + '.txt'
        file = join(gt_path, file)
        key = os.path.split(file)[-1][:-4]
        image_dict[key] = dict()
        image_dict[key]['gt'] = file_to_bbox(file)
        image_dict[key]['preds'] = []

    for file in glob.glob(join(mal_path, '*.txt')):
        key = os.path.split(file)[-1][:-4]
        #print("a",h,w)
        assert key in image_dict
        image_dict[key]['preds'] = file_to_bbox_yolo(file)

    print(len(image_dict.keys()))
    for file in os.listdir(ben_img_path):
    # for file in glob.glob(join(ben_path, '*.txt')):
        if not file.endswith('.png'):
            continue
        file = file[:-4] + '.txt'
        file = join(ben_path, file)
        key = os.path.split(file)[-1][:-4]
        #print("b",h,w)
        assert key not in image_dict
        image_dict[key] = dict()
        image_dict[key]['preds'] = file_to_bbox_yolo(file)
        image_dict[key]['gt'] = []
    # sa = {0:0,0.025:0,0.05:0,0.1:0,0.15:0,0.2:0,0.3:0}
    # for k in image_dict:
    #     conf = image_dict[k]["preds"][0][0] if len(image_dict[k]["preds"]) >0 else 1
    #     if conf > 0:
    #         sa[0] += 1
    #     if conf > 0.025:
    #         sa[0.025] +=1
    #     if conf > 0.05:
    #         sa[0.05] +=1
    #     if conf > 0.1:
    #         sa[0.1] +=1
    #     if conf > 0.15:
    #         sa[0.15]+=1
    #     if conf > 0.2:
    #         sa[0.2] += 1
    #     if conf > 0.3:
    #         sa[0.3] +=1
    # print(sa)
    return image_dict


def pretty_print_fps(senses,fps):
    for s,f in zip(senses,fps):
        print(f'Sensitivity at {f}: {s}')

def get_froc_points(mal_exp, ben_exp, root_fol, fps_req = [0.025,0.05,0.1,0.15,0.2,0.3,0.5,0.7], save_to = None):
    im_dict = generate_image_dict(mal_exp, ben_exp, root_fol = root_fol)
    senses, fps = calc_froc_from_dict(im_dict, fps_req, save_to = save_to)
    return senses, fps

if __name__ == '__main__':
    # seed = '42' if len(sys.argv)== 1 else sys.argv[1]
    mal_exp = sys.argv[1]
    ben_exp = sys.argv[2]
    root_fol = 'runs/detect/'
    save_to = None
    senses, fps = get_froc_points(mal_exp, ben_exp, root_fol, save_to = save_to)

    pretty_print_fps(senses, fps)
