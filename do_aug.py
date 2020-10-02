import augmentation as am
import helpers as hp
import cv2
import os 
import glob
import cv2 as cv
from matplotlib import pyplot as plt

path='./test/*.jpg'
augmentation_dir = "augmentation"

#create the folder augmentation
if not os.path.exists(augmentation_dir):
    os.mkdir(augmentation_dir)

images= glob.glob(path)

for idx, img_path in enumerate(images):
    img_name = img_path.split('/')[-1]
    image = cv2.imread(img_path)
    rd_image = am.random_brightness(image,(-1.3,0.4))
    print("Augmented image: {}".format(os.path.join(augmentation_dir,img_name)))
    cv2.imwrite(os.path.join(augmentation_dir,img_name),rd_image)


# images = hp.load_images(path)
# res_imgs = []
# for img in images:
#     res_imgs.append(am.change_light_by_hsv(img,-1))
# hp.visualize(res_imgs)

# random_brig_imgs = am.random_brightness(images,(0.35,1.5)) 
# hp.visualize(random_brig_imgs, column=3)



# # hp.visualize(images[0:6], column=3, fig_size=(2,3))

# bright_images= am.brighten([img,img2,img3], brightness_coeff=0.9) ## brightness_coeff is between 0.0 and 1.0
# hp.visualize(bright_images, column=3)

# dark_images= am.darken(images[0:6], darkness_coeff=0.6) ## darkness_coeff is between 0.0 and 1.0
# # hp.visualize(dark_images, column=3)


# shadowy_images= am.add_shadow(images[0:6], no_of_shadows=2, shadow_dimension=10) 
# hp.visualize(shadowy_images, column=3)

# images= glob.glob(path)

# img = cv2.imread(images[0])

# brighnees = am.add_shadow(img)

# plt.subplot(1, 2, 1)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# plt.imshow(brighnees)

# plt.show()

# cv2.imshow("out",img)
# # hp.visualize(images, column=3, fig_size=(20,10))
# cv2.waitKey(0)