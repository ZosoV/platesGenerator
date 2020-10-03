import augmentation as am
import helpers as hp
import cv2
import os 
import glob
import cv2 as cv
from matplotlib import pyplot as plt

path='./test/*.jpg'
augmentation_dir = "augmentation"
mini_aug_dir = "mini_aug"

#create the folder augmentation
if not os.path.exists(augmentation_dir):
    os.mkdir(augmentation_dir)

# if not os.path.exists(mini_aug_dir):
#     os.mkdir(mini_aug_dir)

images= glob.glob(path)

for idx, img_path in enumerate(images):
    img_name = img_path.split('/')[-1]
    type_plate = img_name.split('_')[-2][-1]
    random_interval = (-1.28,0.3) if type_plate == "1" else (-1.6,0.3)
    image = cv2.imread(img_path)
    rd_image = am.random_brightness(image,random_interval)
    print("Augmented image: {}  Type: {}".format(os.path.join(augmentation_dir,img_name),type_plate))
    cv2.imwrite(os.path.join(augmentation_dir,img_name),rd_image)


# Proof with little sample of plates

# tipo 1 -1.6
# tipo 2 -1.3

# images= glob.glob(path)

# for idx, img_path in enumerate(images[0:6]):
#     img_name = img_path.split('/')[-1]
    
#     image = cv2.imread(img_path)
#     rd_image = am.change_light_by_hsv(image,0.3)
#     print("Augmented image: {}".format(os.path.join(mini_aug_dir,img_name)))
#     cv2.imwrite(os.path.join(mini_aug_dir,img_name),rd_image)

# random_brig_imgs = am.random_brightness(images,(0.35,1.5)) 
# hp.visualize(random_brig_imgs, column=3)
