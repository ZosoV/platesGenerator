#!/usr/bin/env python

import shutil
import os
import sys

source = "bgs/"
destination = "mini_bgs"
destination = destination + "/" 

if __name__ == "__main__":
    os.mkdir("mini_bgs")
    num_image = int(sys.argv[1])

    for i in range(1,num_image + 1):
        str_m = str(i)
        str_m = str_m.zfill(8)
        shutil.copy(source+str_m+".jpg", destination+str_m+".jpg")
