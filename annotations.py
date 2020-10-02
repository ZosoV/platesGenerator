import argparse
import os
import subprocess
from subprocess import PIPE



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_idx', type=str, default='1',
                        help='Initial index of images to annotate')
    parser.add_argument('--final_idx', type=str, default='5',
                        help='Final idx of the images to annotate')
    parser.add_argument('--img_path', type=str, default='test/'
			help='Select the path of the images'=
    return parser.parse_args()

def writeAnnotations(strNames, strPlates):
    subprocess.run(['touch', 'annotations.txt'])
    annotationsFile = open('annotations.txt', 'w')

    lines = list(map(lambda x, y: 'crops/' + x +'.png ' + y, strNames, strPlates))

    for i in range(0, len(lines)-1):
        annotationsFile.write(lines[i] + '\n')

    annotationsFile.write(lines[-1])

    annotationsFile.close()

def renameFolders(listNames,img_path):
    print(listNames)
    for name in listNames:
        subprocess.run(["cp", img_path +name, "crops/"+name[:12]+".png"])
    


if __name__ == "__main__":
    params = parse_args()
    images = sorted(os.listdir(parser.img_path))
    idx_ini = 0
    idx_fin = 0
    for image in images:
        if str(image[:12]) == params.initial_idx.zfill(12):
            idx_ini = images.index(image)
        if str(image[:12]) == params.final_idx.zfill(12):
            idx_fin = images.index(image)

    print(idx_ini)
    print(idx_fin)
    images = images[idx_ini: idx_fin+1]
    renameFolders(images, params.img_path)

    names = list(map(lambda image: image[:12], images))
    plates = list(map(lambda plate: plate[13:19], images))
    
    

    print(images)
    print(names)
    print(plates)
    writeAnnotations(names, plates)
