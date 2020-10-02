import os
from shutil import copyfile

images_names = os.listdir("mini_bgs")

#create the folder augmentation
if not os.path.exists("tmp_bgs"):
    os.mkdir("tmp_bgs")

for idx, img in enumerate(images_names,1):
    copyfile(os.path.join("mini_bgs",img), \
        os.path.join("tmp_bgs","{:08d}.jpg".format(idx)))

