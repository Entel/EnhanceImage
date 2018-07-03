import os
import shutil
from PIL import Image

PATH = '../data_img/'
IMG_PATH = '../'
FINAL_DATA = '../data'
TMP_PATH = '../tmp'

def rename_img():
    #rename the images and collect them from different director
    i = 1
    j = 1 
    while j <= 6:
        IMG_PATH = str(j) + '/'
        for f in os.listdir(IMG_PATH):
            img_type = f.split('.')[1]
            img = os.path.join(IMG_PATH, f)
            print img
            new_name = str(i) + '.' + img_type
            new_img = os.path.join(PATH, new_name)
            print new_img
            os.rename(img, new_img)
            i += 1
        j += 1

def choose_img():
    #choose the images that can be use as the train data
    for f in os.listdir(PATH):
        path = os.path.join(PATH, f)
        try:
            img = Image.open(path)
            width, height = img.size[0], img.size[1]
            if width >= 960 and height >= 540:
                shutil.copy2(path, FINAL_DATA)
        except:
            print path

def resize_img(path):
    #resize the image close to 960*540
    img = Image.open(path)
    width, height = img.size[0], img.size[1]
    wp = float(width)/960
    hp = float(height)/540
    if wp < hp:
        new_width = 960
        new_height = int(height/wp)
    else:
        new_height = 540
        new_width = int(width/hp)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(path)

def cut_img(path):
    #crop the image into size 960*540
    img = Image.open(path)
    width, height = img.size[0], img.size[1]
    area = (0, 0, 960, 540)
    if width > 960:
        area = ((width-960)/2, 0, (width-960)/2+960, 540)
    elif height > 540:
        area = (0, (height-540)/2, 960, (height-540)/2+540)
    img = img.crop(area)
    img.save(path)

def change_img_format(path):
    #change the format into .png
    try:
        print path
        img = Image.open(path)
        file_name = path.split('data/')[1].split('.')[0]
        new_file_name = file_name + '.png'
        new_file_name = os.path.join(TMP_PATH, new_file_name)
        img.save(new_file_name)
        print '--->' + new_file_name
    except Exception, e:
        print 'Error: ' + path + ' ' + str(e)

#rename_img()
#choose_img()
print 'Finish sellection!'
'''
for f in os.listdir(FINAL_DATA):
    path = os.path.join(FINAL_DATA, f)
    change_img_format(path)
print 'Exchange completed!'
'''
for f in os.listdir(TMP_PATH):
    path = os.path.join(TMP_PATH, f)
    print path
    resize_img(path)
    cut_img(path)
