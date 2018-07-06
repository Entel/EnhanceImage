import os
import shutil
from PIL import Image

PATH = '../data_img/'
IMG_PATH = '../'
FINAL_DATA = '../data'
TMP_PATH = '../data'
FINAL_HEIGHT = 270
FINAL_WIDTH = 480

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
            if width >= FINAL_WIDTH and height >= FINAL_HEIGHT:
                shutil.copy2(path, FINAL_DATA)
        except:
            print path

def resize_img(path):
    #resize the image close to FINAL_WIDTH*FINAL_HEIGHT
    img = Image.open(path)
    width, height = img.size[0], img.size[1]
    wp = float(width)/FINAL_WIDTH
    hp = float(height)/FINAL_HEIGHT
    if wp < hp:
        new_width = FINAL_WIDTH
        new_height = int(height/wp)
    else:
        new_height = FINAL_HEIGHT
        new_width = int(width/hp)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    try:
        img.save(path)
    except Exception, e:
        print 'Error:' + str(e)

def cut_img(path):
    #crop the image into size FINAL_WIDTH*FINAL_HEIGHT
    img = Image.open(path)
    width, height = img.size[0], img.size[1]
    area = (0, 0, FINAL_WIDTH, FINAL_HEIGHT)
    if width > FINAL_WIDTH:
        area = ((width-FINAL_WIDTH)/2, 0, (width-FINAL_WIDTH)/2+FINAL_WIDTH, FINAL_HEIGHT)
    elif height > FINAL_HEIGHT:
        area = (0, (height-FINAL_HEIGHT)/2, FINAL_WIDTH, (height-FINAL_HEIGHT)/2+FINAL_HEIGHT)
    img = img.crop(area)
    try:
        img.save(path)
    except Exception, e:
        print 'Error:' + str(e)

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
'''
choose_img()
print 'Finish sellection!'
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
