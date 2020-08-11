# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import shutil

source = "bgs/"
destination = "mini_bgs"
os.mkdir("mini_bgs")
destination = destination + "/" 

num_image = int(sys.argv[1])

for i in range(1,num_image + 1):
    str_m = str(i)
    str_m = str_m.zfill(12)
    if(not str_m in set_m):
        shutil.copy(source+str_m+".png", destination+str_m+".png")
