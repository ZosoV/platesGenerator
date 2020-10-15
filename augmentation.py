
import cv2 as cv2
import numpy as np
import random
import math

from debug import *

def change_light_contrast(image, coeff, alpha = 1.0):
    new_image = np.zeros(image.shape, image.dtype)
    # alpha = 1.0 # Simple contrast control [1.0 - 3.0]
    beta = coeff * 100    # Simple brightness control [0-100]

    # alpha must be applied to each pixel
    new_image = np.clip(np.multiply(image,alpha) + beta, 0, 255)

    return new_image 

def random_brightness(image,interval = (-1.5,0.5)):
    verify_image(image)

    if(is_list(image)):
        image_RGB=[]
        image_list=image
        for img in image_list:
            random_brightness_coefficient = np.random.uniform(interval[0],interval[1]) ## generates value between 0.0 and 2.0
            image_RGB.append(change_light_contrast(img,random_brightness_coefficient))
    else:
        random_brightness_coefficient = np.random.uniform(interval[0],interval[1]) ## generates value between 0.0 and 2.0
        image_RGB= change_light_contrast(image,random_brightness_coefficient)
    return image_RGB

def shadow_process(image,vertexes,coeff = -0.65, alpha = 1.0):

    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, vertexes, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    
    new_image = np.zeros(image.shape, image.dtype) #new image in zeros
    # alpha = 1.0 # Simple contrast control [1.0 - 3.0]
    beta = coeff * 100    # Simple brightness control [0-100]

    # alpha must be applied to each pixel
    image[mask[:,:,0]==255] = np.clip(np.multiply(image[mask[:,:,0]==255],alpha) + beta, 0, 255)

    return image

def add_shadow(image,var_y_shadow = 0.5, var_bot_x_right = 0, var_bot_x_left = 0, var_top_x_right = 0, var_top_x_left = 0):## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated

    var_y_shadow = np.clip(var_y_shadow,0.0,1.0)
    var_bot_x_right = np.clip(var_bot_x_right,0.0,1.0)
    var_bot_x_left = np.clip(var_bot_x_left,0.0,1.0)

    verify_image(image)

    x1=0       
    y1=0
    x2=image.shape[1]
    #var the max height
    y2=int(image.shape[0] * var_y_shadow) 

    #var the to bottom and top corners
    top_x1 = int(x1 + image.shape[1] * var_top_x_right)
    top_x2 = int(x2 - image.shape[1] * var_top_x_left)

    bot_x1 = int(x1 + image.shape[1] * var_bot_x_right)
    bot_x2 = int(x2 - image.shape[1] * var_bot_x_left)

    #create the vertex of the polygon to be inserted
    vertexes = [(top_x1,y1), (top_x2,y1), (bot_x2,y2), (bot_x1,y2)]
    vertexes = np.array([vertexes], dtype=np.int32)

    if(is_list(image)):
        image_RGB=[]
        image_list=image
        for img in image_list:
            output=shadow_process(img,vertexes)
            image_RGB.append(output)
    else:
        output=shadow_process(image,vertexes)
        image_RGB = output

    return image_RGB

def translation(image, tx, ty):
    #image = cv2.imread(img)
    # Store height and width of the image 
    height, width = image.shape[:2]
    #quarter_height, quarter_width = height / 4, width / 4
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    # We use warpAffine to transform 
    # the image using the matrix, T 
    img_translation = cv2.warpAffine(image, T, (width, height)) 
    return img_translation

def shear(image, tx, ty):
    H, W = image.shape[:2]
#   M2 = np.float32([[1, -0.1, 0], [-0.05, 1, 0]])
    M2 = np.float32([[1, tx, 0], [ty, 1, 0]])
    M2[0,2] = -M2[0,1] * W/2
    M2[1,2] = -M2[1,0] * H/2
    aff2 = cv2.warpAffine(image, M2, (W, H))
    return aff2


