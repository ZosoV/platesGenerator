# License Plates Generator
Algorithm to generate license plates with an specific fonts. The algorithm works in the following way:

1. Take a random background and take a crop from it with the dimensions `OUTPUT_SHAPE`.
2. Generate a license plate image using a random code.
3. Perform an afin transformation of the plate over the background. It helps to
    rotate, and scale the image in a random way.
4. Wrap the transformed plate over the background.
5. Apply a random noise to the final plate.

This algorithm is based in the following [repositroy](https://github.com/matthewearl/deep-anpr) with several changes for our use.

## Use

In this project, you can find different scripts, with different functionality according for generete license plates, usage is as follows:

1. `./extractbgs.py SUN397.tar.gz`: Extract ~3GB of background images from the [SUN database](http://groups.csail.mit.edu/vision/SUN/) into `bgs/`. (`bgs/` must not already exist.) The tar file (36GB) can be [downloaded here](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz). This step may take a while as it will extract 108,634 images.

2. If you can perform some test use `./mini_extract.py 1000`. It extracts a mini dataset of 1000 samples from the complete dataset of `bgs/` in a folder `mini_bgs/` (`mini_bgs/` must not already exist.)

3. `./gen.py --num-img 1000`: Generate 1000 test set images in `test/` using the default settings. (`test/` must not already exist.) This step requires fonts such as `FE-FONT.ttf` or `Helvetica.ttf` to be in the `fonts/` directory.

You can also add some characteristics to `./gen.py` using some parse statements, such as

- `--num-img`: total number of plates generated.
- `--dataset`: folder of the background dataset
- `--font` : a font to draw the plates from the fonts folder
- `--format` : chose the format to return the plate. It can be `1` or `2` for for format 1 or format 2
- `--star-idx` : start to name the images from this index.

4. `./annotation.py --initial_idx 1 final_idx 10`: Label the images from 1 to 10, creating a `annotation.txt` where each line context the following information: `crops/<img_number>.jpg <license_plate_code>` 


### Plates Format

In this project, we use two most known formats of chilean license plates. These format are:

- Format 1: `BB-BB·10` (4 letters y 2 numbers)

	- 18 available letters: B, C, D, F, G, H, J, K, L, P, R, S, T, V, W, X, Y, Z.

	- Available numbers from 10 to 99
	
	- Font: FE-Schrift

- Format 2: `AA·10-00` (2 letters y 4 numbers)
	
	- 23 available letters for the first letter: A, B, C, E, F, G, H, D, K, L, N, P, R, S, T, U, V, X, Y, Z, W y M 
	
    Extra: O for diplomatic representation

	- 23 available letters for the second letter: A, B, C, D, E, F, G, H, I, J, K, L, N, P, R, S, T, U, V, X, Y, Z

	- Available numbers from 1000 to 9999

	- Font: Helvética Medium Condensed



#### Authors

- Oscar Guarnizo
- Diego Suntaxi