import OpenEXR,Imath,os
from PIL import Image
import numpy as np

exr_path = 'F:\\ChengJunyan1\\3d\\ShapeNetCore.v2.OpenEXR'
img_path = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2.Image'
dep_path = 'C:\\ChengJunyan1\\3d\\ShapeNetCore.v2.Depth'

def log(path):
    l_dir=os.listdir(os.path.join(path,'left'))
    r_dir=os.listdir(os.path.join(path,'right'))
    l_log=[i.split('.')[0] for i in l_dir]
    r_log=[i.split('.')[0] for i in r_dir]
    return [l_log,r_log]

def exr2img(exrpath, imgpath):
    file = OpenEXR.InputFile(exrpath)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    rgbf = [Image.frombytes("F", size, file.channel(c, pt)) for c in "RGB"]
    extrema = [im.getextrema() for im in rgbf]
    darkest = min([lo for (lo,hi) in extrema])
    lighest = max([hi for (lo,hi) in extrema])
    scale = 255 / (lighest - darkest)
    def normalize_0_255(v):
        return (v * scale) + darkest
    rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
    img=Image.merge("RGB", rgb8)
    img.save(imgpath)

def exr2dep(exrpath,dep_path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr = OpenEXR.InputFile(exrpath)
    d = exr.channel('Z',pt)
    d = Image.frombytes('F', (320, 320),d)
    d = np.asarray(d)
    d=d*(d<100)
    np.save(dep_path,d)

def img_trans():
    exrl,exrr=log(exr_path)
    exr_l_dir=os.path.join(exr_path,'left')
    exr_r_dir=os.path.join(exr_path,'right')
    logl,logr=log(img_path)
    img_l_dir=os.path.join(img_path,'left')
    img_r_dir=os.path.join(img_path,'right')
    for i in exrl:
        name=i.split('+')[0]+'+'+i.split('+')[1]
        if name not in logl:
            exr2img(os.path.join(exr_l_dir,i+'.exr'),os.path.join(img_l_dir,name+'.png'))
            print(name+' left img saved')
        if name not in logr:
            exr2img(os.path.join(exr_r_dir,i+'.exr'),os.path.join(img_r_dir,name+'.png'))
            print(name+' right img saved\n')
                  
def dep_trans():
    exrl,exrr=log(exr_path)
    exr_l_dir=os.path.join(exr_path,'left')
    logl,logr=log(dep_path)
    dep_l_dir=os.path.join(dep_path,'left')
    for i in exrl:
        name=i.split('+')[0]+'+'+i.split('+')[1]
        if name not in logl:
            exr2dep(os.path.join(exr_l_dir,i+'.exr'),os.path.join(dep_l_dir,name))
            print(name+' left dep saved')