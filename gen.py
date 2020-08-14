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
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import argparse

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized
PARAMS = None

#OUTPUT_SHAPE = (64, 128)
OUTPUT_SHAPE = (24, 94)

#SPECIFICATIONS OF THE PLATES FORMATS

#FORMAT 1 - BB BB 00 
#where the B can get a letter in LETTER_FORMAT_1 and 0 can be any digit from 0 to 9
LETTER_FORMAT_1 = 'BCDFGHJKLPRSTVWXYZ' 

#FORMAT 2 - AB 00 00 
#where the A can get a letter in LETTER_1_FORMAT_2, and B can get a letter in LETTER_2_FORMAT_2
#and 0 can be any digit from 0 to 9
LETTER_1_FORMAT_2 = 'ABCDEFGHKLNPRSTUVXYZWM'
LETTER_2_FORMAT_2 = 'ABCDEFGHIJKLNPRSTUVXYZ'

#Variables to create the char using a font
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS
CHARS = CHARS + " "


#Function that returns a char with a its respective img RGB using a font
def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


#Function that apply different rotation to the plate
def euler_to_mat(rotate_x, rotate_y, rotate_z):
    # Rotate clockwise about the X-axis
    c, s = math.cos(rotate_x), math.sin(rotate_x)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]])

    # Rotate clockwise about the Y-axis
    c, s = math.cos(rotate_y), math.sin(rotate_y)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(rotate_z), math.sin(rotate_z)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


#Function that take the color of the plates
def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


#Function that creates a M matrix for a affine transformation
def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          rotation_variation=0.9):

    #Define the shape as size. The difference is the order and type.
    #shape is (h,w) and size is (w,h) in numpy.array                      
    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

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
    M = numpy.hstack([M, center_to - M * center_from])

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
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out

# Function that generates the images of the plate
def generate_plate(font_height, char_ims, extra_spacing):
    #Define some spacing in the vertical and horizontal way
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height

    #Define the spacing between characters
    spacing = font_height * random.uniform(-0.05, 0.05)
    
    #Definig the radious of the rounded rect of the plate
    radius = 1 + int(font_height * 0.1 * random.random())

    #Generate a random code to a plate
    code = generate_code()

    #Define the text_widh considering the space of each letter and
    #some spacing between them
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    #Here we use extra spacing to follow the format 1 and 2 of chilean license plates
    #The extra spacing is added each two characters
    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + (h_padding * 2) + extra_spacing[0] + extra_spacing[1]))

    #Take the text color and plate color
    text_color, plate_color = pick_colors()
    
    #Generate the matrix where we will place the character of the plate
    text_mask = numpy.zeros(out_shape)
    
    #Iterate for the characters of code adding some padding, spacing a extra spacing 
    #between character.
    x = h_padding
    y = v_padding 

    i = 0
    for idx, c in enumerate(code,1):
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing
        if (idx % 2 == 0 and idx < len(code)):
            x += extra_spacing[i]
            i += 1

    #Place the color of text and plate in the final plate
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    '''
    Takes randomly a background image and selects the first 
    with size greater than the OUTPUT_SHAPE. Then crops this image
    in a random region with the dimension of OUTPUT_SHAPE
    '''    
    found = False
    while not found:
        fname = PARAMS.dataset + "/{:08d}.jpg".format(random.randint(1, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True
    #take a random crop of the original background
    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1]) 
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0]) 
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]] #bg[top:bottom,left:rigth]

    return bg


def generate_im(font_char_ims, num_bg_images):
    '''
    Generate images with a background and a plate given

    font_char_ims: the dict of the char with a especific font
    num_bg_image: the total number of images generated
    '''
    #Generate a background given the total number of background
    bg = generate_bg(num_bg_images)

    #Control the spacing each two characters based on format 1 and format 2
    extra_spacing = []
    if (PARAMS.format == 1):
        extra_spacing = [3.5, 6.5]
    elif (PARAMS.format == 2):
        extra_spacing = [7.5, 2.5]

    #Generate a plate and plate_mask given a height and the dictionary of fonts
    plate, plate_mask, code = generate_plate(FONT_HEIGHT, font_char_ims, extra_spacing)
    
    #Return the matrix M to perform the affine transfromation with the plate into the background
    #out_of_bound is a boolean that indicates if the plate is out of bound of the background.
    M = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.45,
                            max_scale=0.55,
                            rotation_variation=0.2)

    #Creating the plate and plate_mask with a size given by bg              
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    #Joining the plate and plate_mask and adding the background
    out = plate * plate_mask + bg * (1.0 - plate_mask)

    #Resize the image with the final outputs
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    #Adding Noise
    out += numpy.random.normal(scale=0.085, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code


# Function that create the dictionary of fonts
def load_fonts(folder_path,font):
    '''
    Creates a dictionary of the given font

    :return:
        Dictionary: key - character, value - the matrix of character
    '''
    font_char_ims = {}

    font_char_ims = dict(make_char_ims(os.path.join(folder_path,
                                                font),
                                                FONT_HEIGHT))
    return font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """

    #get the dictionary of fonts where key is the character and value are the matrix of that character
    font_char_ims = load_fonts(FONT_DIR,PARAMS.font)

    #takes the total number of background in the folder PARAMS.dataset
    num_bg_images = len(os.listdir(PARAMS.dataset))
    while True:
        #generate an image given the dictionaty of fonts and the total number of background
        yield generate_im(font_char_ims, num_bg_images)


#Function to take the params by command line
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-img', type=int, default=10,
                        help='Number of generated images')
    parser.add_argument('--font', type=str, default="FE-FONT.ttf",
                        help='Chose the font of the plates')
    parser.add_argument('--format', type=int, default=1,
                        help='Chose the correct format 1: current and 2: past format')
    parser.add_argument('--star-idx', type=int, default=1,
                        help='Chose the index to start the names of images')
    parser.add_argument('--dataset', type=str, default="mini_bgs",
                        help='Chose the dataset to generate the images')
    return parser.parse_args()


def main():
    #create the folder test
    os.mkdir("test")

    #create a iterator with coroutine that five the total number of images
    im_gen = itertools.islice(generate_ims(), PARAMS.num_img)

    #iterate through the iterator display the name and store in the folder test
    for img_idx, (im, c) in enumerate(im_gen,PARAMS.star_idx):
        fname = "test/{:012d}_{}.png".format(img_idx,c)
        print(fname)
        cv2.imwrite(fname, im * 255.)

if __name__ == "__main__":

    #Take the argument by command line
    PARAMS = parse_args()

    main()

