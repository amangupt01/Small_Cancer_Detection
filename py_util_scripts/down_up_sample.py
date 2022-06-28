import os
import sys
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

for scale in [2,4]:
  root = f'Small_Mass_Validation/{scale}k_up/test'

  for type in ['mal', 'ben']:
    im_folder_path = f'{root}/{type}/images'
    for img_name in tqdm(os.listdir(im_folder_path)):
      im_path = f'{im_folder_path}/{img_name}'
      img = Image.open(im_path)

      scaled_shape = (int(img.height / scale), int(img.width / scale))
      img_down = transforms.functional.resize(img, scaled_shape)

      scaled_shape = (img.height, img.width)
      img_up = transforms.functional.resize(img_down, scaled_shape)

      img_up.save(im_path)


