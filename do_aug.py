import augmentation as am
import helpers as hp
import cv2
import os 
import glob
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD # Communicador de MPI 
size = comm.Get_size() # Numero total de Procesadores
rank = comm.Get_rank() # Id de cada procesador

path='./test/*.jpg'
augmentation_dir = "gen_augmentation"
mini_aug_dir = "mini_aug"

#create the folder augmentation
if not os.path.exists(augmentation_dir) and rank == 0:
    os.mkdir(augmentation_dir)

images= glob.glob(path)

def brigth_aug_chunk(image_list):
    print("AUGMENTATION BRIGTHNESS AND CONTRAST")

    # Brigthness and Contrast # Augemented
    for idx, img_path in enumerate(image_list):

        # string name
        img_name = img_path.split('/')[-1]
        type_plate = img_name.split('_')[-2][-1]
        aug_name = img_name.split('.')[-2] + "_aug_1" + ".jpg"

        #load image
        image = cv2.imread(img_path)

        # define the random parameter
        random_interval = (-1.28,0.3) if type_plate == "1" else (-1.6,0.3)
        coeff_brightness = np.random.uniform(random_interval[0],random_interval[1])

        #agumentation
        rd_image = am.change_light_contrast(image,coeff_brightness)

        #save image
        print("Augmented image: {} Augmentation Type: {} Type: {}".format(os.path.join(augmentation_dir,img_name),"1",type_plate))
        cv2.imwrite(os.path.join(augmentation_dir,aug_name),rd_image)

def random_shadow_aug_chunk(image_list):
    print("AUGMENTATION RANDOM SHADOW")
    # Random Shadow # Augemented 3
    for idx, img_path in enumerate(image_list):

        # string name
        img_name = img_path.split('/')[-1]
        type_plate = img_name.split('_')[-2][-1]
        aug_name = img_name.split('.')[-2] + "_aug_3" + ".jpg"

        #load image
        image = cv2.imread(img_path)

        # define the random parameter
        random_type = np.random.choice([1,2,3])

        rd_image = None

        if random_type == 1:
            rnd_var_y_shadow = np.random.uniform(0,0.5)
            #agumentation
            rd_image = am.add_shadow(image,var_y_shadow=rnd_var_y_shadow)

        elif random_type == 2:
            rnd_var_y_shadow = np.random.uniform(0.9,1.0)
            rnd_var_bot_x_right = np.random.uniform(0.35,0.45)
            rnd_var_top_x_right = np.random.uniform(0.0,0.25)
            #agumentation
            rd_image = am.add_shadow(image,
                var_y_shadow = rnd_var_y_shadow,
                var_bot_x_right = rnd_var_bot_x_right,
                var_top_x_right = rnd_var_top_x_right)

        elif random_type == 3:
            rnd_var_y_shadow = np.random.uniform(0.9,1.0)
            rnd_var_bot_x_left = np.random.uniform(0.35,0.45)
            rnd_var_top_x_left = np.random.uniform(0.0,0.25)
            #agumentation
            rd_image = am.add_shadow(image,
                var_y_shadow = rnd_var_y_shadow,
                var_bot_x_left = rnd_var_bot_x_left,
                var_top_x_left = rnd_var_top_x_left)

        # define the random parameter
        random_interval = (-0.78,0.3) if type_plate == "1" else (-1.1,0.3)
        coeff_brightness = np.random.uniform(random_interval[0],random_interval[1])

        #agumentation brigthness
        rd_image = am.change_light_contrast(rd_image,coeff_brightness)

        #save image
        print("Augmented image: {} Augmentation Type: {} Format Type: {}".format(os.path.join(augmentation_dir,img_name),"3",type_plate))
        cv2.imwrite(os.path.join(augmentation_dir,aug_name),rd_image)


def shear_bright_aug_chunk(image_list):
    print("AUGMENTATION OF SHEAR")
    # Brigthness and Contrast # Augemented
    for idx, img_path in enumerate(image_list):

        # string name
        img_name = img_path.split('/')[-1]
        type_plate = img_name.split('_')[-2][-1]
        aug_name = img_name.split('.')[-2] + "_aug_2" + ".jpg"

        #load image
        image = cv2.imread(img_path)

        # define the random parameter
        random_interval = (-1.28,0.3) if type_plate == "1" else (-1.6,0.3)
        coeff_brightness = np.random.uniform(random_interval[0],random_interval[1])
        tx = np.random.uniform(-0.1,0.1)
        ty = np.random.uniform(-0.05,0.05)
        

        #agumentation
        rd_image = am.change_light_contrast(image,coeff_brightness)
        sh_image = am.shear(rd_image, tx, ty)

        #save image
        print("Augmented image: {} Augmentation Type: {} Type: {}".format(os.path.join(augmentation_dir,img_name),"2",type_plate))
        cv2.imwrite(os.path.join(augmentation_dir,aug_name),sh_image)

        
def shear_bright_tras_aug_chunk(image_list):
    print("AUGMENTATION RANDOM TRASLATION")
    # Brigthness and Contrast # Augemented
    for idx, img_path in enumerate(image_list):

        # string name
        img_name = img_path.split('/')[-1]
        type_plate = img_name.split('_')[-2][-1]
        aug_name = img_name.split('.')[-2] + "_aug_4" + ".jpg"

        #load image
        image = cv2.imread(img_path)

        # define the random parameter
        random_interval = (-1.28,0.3) if type_plate == "1" else (-1.6,0.3)
        coeff_brightness = np.random.uniform(random_interval[0],random_interval[1])
        tx = np.random.uniform(-0.1,0.1)
        ty = np.random.uniform(-0.05,0.05)

        tx_tras = np.random.uniform(-5,5)
        

        #agumentation
        rd_image = am.change_light_contrast(image,coeff_brightness)
        sh_image = am.shear(rd_image, tx, ty)
        tras_image = am.translation(sh_image, tx_tras, 0)

        #save image
        print("Augmented image: {} Augmentation Type: {} Type: {}".format(os.path.join(augmentation_dir,img_name),"4",type_plate))
        cv2.imwrite(os.path.join(augmentation_dir,aug_name),tras_image)        


chunks_size = len(images) // 32

if (rank == 0): brigth_aug_chunk(images[0:chunks_size])
if (rank == 1): brigth_aug_chunk(images[chunks_size:chunks_size*2])
if (rank == 2): brigth_aug_chunk(images[chunks_size*2:chunks_size*3])
if (rank == 3): brigth_aug_chunk(images[chunks_size*3:chunks_size*4])
if (rank == 4): brigth_aug_chunk(images[chunks_size*4:chunks_size*5])
if (rank == 5): brigth_aug_chunk(images[chunks_size*5:chunks_size*6])
if (rank == 6): brigth_aug_chunk(images[chunks_size*6:chunks_size*7])
if (rank == 7): brigth_aug_chunk(images[chunks_size*7:chunks_size*8])

if (rank == 8): shear_bright_aug_chunk(images[chunks_size*8:chunks_size*9])
if (rank == 9): shear_bright_aug_chunk(images[chunks_size*9:chunks_size*10])
if (rank == 10): shear_bright_aug_chunk(images[chunks_size*10:chunks_size*11])
if (rank == 11): shear_bright_aug_chunk(images[chunks_size*11:chunks_size*12])
if (rank == 12): shear_bright_aug_chunk(images[chunks_size*12:chunks_size*13])
if (rank == 13): shear_bright_aug_chunk(images[chunks_size*13:chunks_size*14])
if (rank == 14): shear_bright_aug_chunk(images[chunks_size*14:chunks_size*15])
if (rank == 15): shear_bright_aug_chunk(images[chunks_size*15:chunks_size*16])

if (rank == 16): random_shadow_aug_chunk(images[chunks_size*16:chunks_size*17])
if (rank == 17): random_shadow_aug_chunk(images[chunks_size*17:chunks_size*18])
if (rank == 18): random_shadow_aug_chunk(images[chunks_size*18:chunks_size*19])
if (rank == 19): random_shadow_aug_chunk(images[chunks_size*19:chunks_size*20])
if (rank == 20): random_shadow_aug_chunk(images[chunks_size*20:chunks_size*21])
if (rank == 21): random_shadow_aug_chunk(images[chunks_size*21:chunks_size*22])
if (rank == 22): random_shadow_aug_chunk(images[chunks_size*22:chunks_size*23])
if (rank == 23): random_shadow_aug_chunk(images[chunks_size*23:chunks_size*24])

if (rank == 24): shear_bright_tras_aug_chunk(images[chunks_size*24:chunks_size*25])       
if (rank == 25): shear_bright_tras_aug_chunk(images[chunks_size*25:chunks_size*26])       
if (rank == 26): shear_bright_tras_aug_chunk(images[chunks_size*26:chunks_size*27])       
if (rank == 27): shear_bright_tras_aug_chunk(images[chunks_size*27:chunks_size*28])       
if (rank == 28): shear_bright_tras_aug_chunk(images[chunks_size*28:chunks_size*29])       
if (rank == 29): shear_bright_tras_aug_chunk(images[chunks_size*29:chunks_size*30])       
if (rank == 30): shear_bright_tras_aug_chunk(images[chunks_size*30:chunks_size*31])       
if (rank == 31): shear_bright_tras_aug_chunk(images[chunks_size*31:-1])  
