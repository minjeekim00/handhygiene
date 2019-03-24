import os
import imageio
import numpy as np
from glob import glob

def generate_gif(dir, size=None):
    """
        argument: (dir): directory of image frames
    """
    imagefiles = glob(os.path.join(dir, '*.jpg'))
    if len(imagefiles) == 0:
        imagefiles = glob(os.path.join(dir, '*.png'))
        
    images = []
    for filename in sorted(imagefiles):
        if size:
            from PIL import Image
            image = Image.open(filename)
            image = np.asarray(image.resize(size))
        else:
            image = imageio.imread(filename)
        images.append(image)
    basename = os.path.basename(dir)
    imageio.mimsave(os.path.join(dir, '{}_rgb.gif'.format(basename)), images)
    
    
    
def generate_gif_from_npy(npyfile):
    """
        argument: (npyfile): .npy file of flow array
    """
    npy = np.load(npyfile)
    images = []
    for image in npy:
        shape = image.shape
        image = image.astype(np.uint8)
        if shape[-1] == 2:
            ch3 = np.ones((shape[0], shape[1], 1), dtype=np.uint8)*128
            images.append(np.dstack((image, ch3)))
        else:
            images.append(image)
    dirname = os.path.dirname(npyfile)
    basename = os.path.basename(dirname)
    dstname = '{}_rgb.gif'.format(basename) if npy.shape[-1] != 2 else '{}_flow.gif'.format(basename)
    imageio.mimsave(os.path.join(dirname, dstname), images)

    
    
    
    
