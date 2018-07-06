import os
from PIL import Image

PATH = '../data'
FINAL_PATH = '../data/'

SIZE_1 = (160, 90)
SIZE_2 = (480, 270)
def resize_an_img(path):
    #1. resize the image into 480*270
    #2. resize the image into 1920*1080
    img = Image.open(path)
    img = img.resize(SIZE_1, Image.ANTIALIAS)
    img = img.resize(SIZE_2, Image.ANTIALIAS)
    full_name = path.split('data/')[1]
    file_name, file_format = full_name.split('.')[0], full_name.split('.')[1]
    new_file_name = file_name + '_.' + file_format
    print new_file_name
    img.save(os.path.abspath(FINAL_PATH + new_file_name))

for f in os.listdir(PATH):
    path = os.path.join(PATH, f)
    resize_an_img(path)

#resize_an_img('13.png')
