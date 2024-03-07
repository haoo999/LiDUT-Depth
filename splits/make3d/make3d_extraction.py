# 获得make3d图片的名称
 
import glob
imglist = glob.glob('../../datasets/make3d/Test134/*.jpg')
 
with open('make3d_test_files.txt', 'a+') as f:
    for i in imglist:
        _, _, _, _, _, _, name = i.split('/')
        f.write('{}\n'.format(name[:-4]))