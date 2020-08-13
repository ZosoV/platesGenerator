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

import common
import argparse

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized
PARAMS = None

#OUTPUT_SHAPE = (64, 128)
OUTPUT_SHAPE = (24, 94)


CHARS = common.CHARS + " "


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


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          rotation_variation=0.9):

    #Define the shape as size. The difference is the order and type.
    #shape is (h,w) and size is (w,h) in numpry.array                      
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


def generate_code():
    if (PARAMS.format == 1):
        return "{}{}{}{}{}{}".format(
                random.choice(common.actual_format),
                random.choice(common.actual_format),
                random.choice(common.actual_format),
                random.choice(common.actual_format),
                random.choice(common.DIGITS),
                random.choice(common.DIGITS))
    else:
        return "{}{}{}{}{}{}".format(
                random.choice(common.past_format_letter1),
                random.choice(common.past_format_letter2),
                random.choice(common.DIGITS),
                random.choice(common.DIGITS),
                random.choice(common.DIGITS),
                random.choice(common.DIGITS))


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


def generate_plate(font_height, char_ims, extra_spacing):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + (h_padding * 2) + extra_spacing[0] + extra_spacing[1]))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
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
        fname = "mini_bgs/{:08d}.jpg".format(random.randint(1, num_bg_images))
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1]) #take a random crop of the original background
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0]) #take a random crop of the original background
    #[top:bottom,left:rigth]
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

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
                            min_scale=0.55,
                            max_scale=0.63,
                            rotation_variation=0.7)

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

    #takes the total number of background in the folder mini_bgs
    num_bg_images = len(os.listdir("mini_bgs"))
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
    parser.add_argument('--star-idx', type=int, default=5602,
                        help='Chose the index to start the names of images')
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

