Trained Weights are present on the [link](https://drive.google.com/drive/folders/1xveBXl2mj7Ktrv03qs-YTiOrtJfAk76s?usp=sharing). The weights of best performing model are present in the ```exp120/``` folder.  

Reproducability of Best Perfoming Model:
   * **Resizing** - Assuming we have input annotated dataset with mamogram 4k images, first we need to resize the images to 1k & 2k resolution using the Pytorch/OpenCv standard utils with Bilinear Interpolation. Use this [resize script](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/py_util_scripts/convert_gt.py) to corresponding resize groundtruth annotations. 
   * **Systematic Crop** - We crop images of 2k and 4k in fragments of size 1k and corresponding create annotations for each fragment. Use this [crop script](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/crop_scripts/systematic_crop.py), You might need to change the path to your own dataset.
   * **Training the Model** - Configure the file [coco128.yml](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/data/coco.yaml) to add paths for the datasets, it should include 1k images and 1k crops from 2k & 4k images. Run [train.py](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/train.py)
   * **Testing at three scales** - Test the model obtained in the previous step on 1k, 2k & 4k resolution datasets respectively using [detect.py](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/detect.py). 
   * **Merging the obtained predictions** - Merge the predictions obtained in the previous step using the script [merge_wbf.py](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/merge_scripts/merge_wbf.py)
   * **Evaluation** - To get the values FPI at various Sensitivities use [FROC.py](https://github.com/amangupt01/Small_Mass_Detection_Yolov5/blob/master/froc_scripts/FROC_merge.py)
        
Credits:  
   * Used the base implementation of Yolov5 from https://github.com/ultralytics/yolov5
   * Wbf script for merging is taken from https://github.com/ZFTurbo/Weighted-Boxes-Fusion
