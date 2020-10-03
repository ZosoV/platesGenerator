#!/usr/bin/env python

"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy as np

import augmentation as am

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import argparse

FONT_DIR = "./fonts"
FONT_HEIGHT = [32,34]  # Pixel size to which the chars are resized
                       # Type 1 , Type 2
PARAMS = None

#OUTPUT_SHAPE = (64, 128)
OUTPUT_SHAPE = (24, 94)

#COLORS OF PLATES
COLOR_PROBABILITIES = [0.65, 0.25, 0.10]
COLOR_NAMES = ["blanco","naranja","amarillo"]
PLATES_COLORS = [(255,255,255), (12,88,241), (36,167,241)] 
COLOR_ID = 0

#Noise probability
NOISE_PROBABILITY = [0.5, 0.15, 0.35]
NOISES = ["gauss", "s&p", "poisson"]
NOISE_ID = 0

#Plate Variations
MIN_SCALE=0.4
MAX_SCALE=0.50
ROTATION_VARIATION=0.25

#Type Configuration
PLATE_MARGINS = [ 
    {
        "h_padding" : 0.2,
        "top_padding" : 0.2,
        "bottom_padding" : 0.62,
        "spacing" : 0.005,
        "extra_spacing": [0.2, 0.4],
        "min_scale": 0.38,
        "max_scale": 0.48 
        },
    {
        "h_padding" : 0.1,
        "top_padding" : 0.005,
        "bottom_padding" : 0.42,
        "spacing" : 0.005,
        "extra_spacing": [0.2, 0.08],
        "min_scale": 0.5,
        "max_scale": 0.55 
        }]


#SPECIFICATIONS OF THE PLATES FORMATS

#FORMAT 1 - BB BB 00 
#where the B can get a letter in LETTER_FORMAT_1 and 0 can be any digit from 0 to 9
LETTER_FORMAT_1 = 'BCDFGHJKLPRSTVWXYZ' 

#FORMAT 2 - AB 00 00 
#where the A can get a letter in LETTER_1_FORMAT_2, and B can get a letter in LETTER_2_FORMAT_2
#and 0 can be any digit from 0 to 9
LETTER_1_FORMAT_2 = 'ABCDEFGHKLNPRSTUVXYZWM'
LETTER_2_FORMAT_2 = 'ABCDEFGHIJKLNPRSTUVXYZW'

#Variables to create the char using a font
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS
CHARS = CHARS + " "


#Function that returns a char with a its respective img RGB using a font
def make_char_ims(font_path, output_height, color):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGB", (width, height), color)

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (0, 0, 0), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im).astype(np.uint8)

#Function that apply different rotation to the plate
def euler_to_mat(rotate_x, rotate_y, rotate_z):
    # Rotate clockwise about the X-axis
    c, s = math.cos(rotate_x), math.sin(rotate_x)
    M = np.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]])

    # Rotate clockwise about the Y-axis
    c, s = math.cos(rotate_y), math.sin(rotate_y)
    M = np.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(rotate_z), math.sin(rotate_z)
    M = np.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M

#Function that creates a M matrix for a affine transformation
def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          rotation_variation=0.9):

    #Define the shape as size. The difference is the order and type.
    #shape is (h,w) and size is (w,h) in np.array                      
    from_size = np.array([[from_shape[1], from_shape[0]]]).T
    to_size = np.array([[to_shape[1], to_shape[0]]]).T

    #Select a random scale between the interval
    scale = random.uniform(min_scale,max_scale)
                           
    #Define the random rotation in each axis
    rotate_z = random.uniform(-0.3, 0.3) * rotation_variation
    rotate_x = random.uniform(-0.2, 0.2) * rotation_variation
    rotate_y = random.uniform(-1.2, 1.2) * rotation_variation

    #Take the center of the background holder and the plate holder
    center_to = to_size / 2.
    center_from = from_size / 2.

    #Evaluate the ratotations and take the matrix
    M = euler_to_mat(rotate_x, rotate_y, rotate_z)[:2, :2]

    #Scale the image
    M *= scale

    #Center the plate holder
    M = np.hstack([M, center_to - M * center_from])

    return M

#Function that generates de code of plate using a prefined format
def generate_code():
    if (PARAMS.format == 1):
        return "{}{}{}{}{}{}".format(
                random.choice(LETTER_FORMAT_1),
                random.choice(LETTER_FORMAT_1),
                random.choice(LETTER_FORMAT_1),
                random.choice(LETTER_FORMAT_1),
                random.choice(DIGITS),
                random.choice(DIGITS))
    else:
        return "{}{}{}{}{}{}".format(
                random.choice(LETTER_1_FORMAT_2),
                random.choice(LETTER_2_FORMAT_2),
                random.choice(DIGITS),
                random.choice(DIGITS),
                random.choice(DIGITS),
                random.choice(DIGITS))


#Function that return a rounded rect given a shape and radius
def rounded_rect(shape, radius):
    out = np.ones(shape,dtype=np.uint8)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    # Change the out in 3 channels
    out_3c = np.zeros( ( np.array(out).shape[0], np.array(out).shape[1], 3 ),dtype=np.uint8)
    out_3c[:,:,0] = out * 255
    out_3c[:,:,1] = out * 255
    out_3c[:,:,2] = out * 255

    return out_3c

def overlap_mask(img1,img2, mask_overlay=None):

    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    mask = None
    if (mask_overlay is None):
        _ , mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    else:
        mask_overlay = cv2.cvtColor(mask_overlay,cv2.COLOR_BGR2GRAY)
        _ , mask = cv2.threshold(mask_overlay, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows, 0:cols ] = dst

    return img1

# Function that generates the images of the plate
def generate_plate(font_height, font_char_ims):

    #Select the char of an specific color and the color
    random_idx = np.random.choice(len(PLATES_COLORS),p=COLOR_PROBABILITIES)
    char_ims = font_char_ims[random_idx]
    plate_color = PLATES_COLORS[random_idx]
    print("COLOR: {}".format(COLOR_NAMES[random_idx]))

    #Select the info parameter according to the type of plate
    plate_info = PLATE_MARGINS[PARAMS.format - 1]

    #Define some spacing in the vertical and horizontal way
    h_padding = plate_info["h_padding"] * font_height
    top_padding = plate_info["top_padding"] * font_height
    bottom_padding = plate_info["bottom_padding"] * font_height

    #Define the spacing between characters
    spacing = font_height * plate_info["spacing"]
    
    #Definig the radious of the rounded rect of the plate
    radius = 1 + int(font_height * 0.1)

    #Generate a random code to a plate
    code = generate_code()

    #Define the text_widh considering the space of each letter and
    #some spacing between them
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    #Here we use extra spacing to follow the format 1 and 2 of chilean license plates
    #The extra spacing is added each two characters
    out_shape = (int(font_height + top_padding + bottom_padding) ,
                 int(text_width + (h_padding * 2) + plate_info["extra_spacing"][0] * font_height + plate_info["extra_spacing"][1] * font_height))
    
    #Generate the matrix where we will place the character of the plate
    # text_mask = np.zeros(out_shape)

    text_mask = np.zeros((out_shape[0],out_shape[1],3),dtype=np.uint8)
    text_mask[:]=plate_color

    #Iterate for the characters of code adding some padding, spacing a extra spacing 
    #between character.
    x = h_padding
    y = top_padding

    i = 0
    for idx, c in enumerate(code,1):
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing
        if (idx % 2 == 0 and idx < len(code)):
            x += plate_info["extra_spacing"][i] * font_height
            i += 1

    # Png image mask
    png_mask = cv2.imread('mask/mask_type1.png') if (PARAMS.format == 1) else \
        cv2.imread('mask/mask_type2_no_symbols.png')

    resize_mask = cv2.resize(png_mask, (text_mask.shape[1],text_mask.shape[0]))

    plate_with_mask = overlap_mask(text_mask,resize_mask)

    # cv2.imshow("out",plate_with_mask)
    # cv2.waitKey(0)

    return plate_with_mask, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    '''
    Takes randomly a background image and selects the first 
    with size greater than the OUTPUT_SHAPE. Then crops this image
    in a random region with the dimension of OUTPUT_SHAPE
    '''    
    found = False
    while not found:
        fname = PARAMS.dataset + "/{:08d}.jpg".format(random.randint(1, num_bg_images))
        bg = cv2.imread(fname)
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True
    #take a random crop of the original background
    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1]) 
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0]) 
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]] #bg[top:bottom,left:rigth]

    return bg

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = 2
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy

def generate_im(font_char_ims, num_bg_images):
    '''
    Generate images with a background and a plate given

    font_char_ims: the dict of the char with a especific font
    num_bg_image: the total number of images generated
    '''
    #Generate a background given the total number of background
    bg = generate_bg(num_bg_images)

    #Generate a plate and plate_mask given a height and the dictionary of fonts
    plate, plate_mask, code = generate_plate(FONT_HEIGHT[PARAMS.format-1], font_char_ims)
    
    #Return the matrix M to perform the affine transfromation with the plate into the background
    #out_of_bound is a boolean that indicates if the plate is out of bound of the background.
    M = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=PLATE_MARGINS[PARAMS.format-1]["min_scale"],
                            max_scale=PLATE_MARGINS[PARAMS.format-1]["max_scale"],
                            rotation_variation=ROTATION_VARIATION)

    #Creating the plate and plate_mask with a size given by bg              
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    #Adding the plate to the background
    out = overlap_mask(bg,plate,plate_mask)

    #Resize the image with the final outputs
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    #Select Random Noise
    random_noise = np.random.choice(NOISES,p = NOISE_PROBABILITY)
    print("NOISE: {}".format(random_noise))

    #Adding Noise
    out = noisy(random_noise,out)
    # out = np.clip(out, 0., 1.)

    return out, code


# Function that create the dictionary of fonts
def load_fonts(folder_path,font):
    '''
    Creates a dictionary of the given font

    :return:
        Dictionary: key - character, value - the matrix of character
    '''
    font_char_ims = []

    for color in PLATES_COLORS:
        font_char_ims.append(dict(make_char_ims(os.path.join(folder_path,
                                                font),
                                                FONT_HEIGHT[PARAMS.format - 1],color)))
    return font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """

    #get the dictionary of fonts where key is the character and value are the matrix of that character
    
    font = "FE-FONT.ttf" if PARAMS.format == 1 else "Helvetica.ttf"
    font_char_ims = load_fonts(FONT_DIR,font)

    #takes the total number of background in the folder PARAMS.dataset
    num_bg_images = len(os.listdir(PARAMS.dataset))
    while True:
        #generate an image given the dictionaty of fonts and the total number of background
        yield generate_im(font_char_ims, num_bg_images)


#Function to take the params by command line
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-img', type=int, default=5,
                        help='Number of generated images')
    parser.add_argument('--format', type=int, default=1,
                        help='Chose the correct format 1: current and 2: past format')
    parser.add_argument('--star-idx', type=int, default=12951,
                        help='Chose the index to start the names of images')
    parser.add_argument('--dataset', type=str, default="mini_bgs",
                        help='Chose the dataset of backgrounds to generate the images')
    return parser.parse_args()


def main():
    #create the folder test
    if not os.path.exists("test"):
        os.mkdir("test")

    #create a iterator with coroutine that five the total number of images
    im_gen = itertools.islice(generate_ims(), PARAMS.num_img)

    #iterate through the iterator display the name and store in the folder test
    for img_idx, (im, c) in enumerate(im_gen,PARAMS.star_idx):
        fname = "test/{:012d}-gen{}_{}.jpg".format(img_idx,PARAMS.format,c)
        print(fname)
        cv2.imwrite(fname, im)

if __name__ == "__main__":

    #Take the argument by command line
    PARAMS = parse_args()

    main()

