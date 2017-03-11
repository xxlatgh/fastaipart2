from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import PIL 

import importlib
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import scipy.ndimage.filters

from keras import metrics
from vgg16_avg import VGG16_Avg

def img_resize(img):
    '''Resize the image to less than 500 x 500. 
        Args:
            img: an actual image.          
    '''    
    if (img.size[0]>500) or (img.size[1]>500):
        factor = (img.size[0]+img.size[1])/1000
        img = img.resize(np.divide(img.size, factor).astype('int32'))
    return img


def img_norm(img_arr):
    '''Normalize an image np array against imagenet mean and return as np array.
        Args:
            img_arr: a numpy array of an image.
    '''
    imagenet_mean = [123.68, 116.779, 103.939]
    rn_mean = np.array((imagenet_mean), dtype=np.float32)   
    img_arr = (img_arr- rn_mean)[:, :, :, ::-1] # Flip the channels from RGB to BGR
    return img_arr

def load_image(size, path):
    '''Load image from disc and return as np array.
        Args:
            size: The desired dimensions of the output image.
            path: The full path of the image to load.
    '''
    width, height = size
    img = Image.open(path)
    img = img.resize((width, height))
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, 0)
    return img_arr

def get_content(size, content_path):
    '''Get content image from disc and return as a np array.
        Args:
            size: The desired dimensions of the output image.
            content_path: The full path of the content image to load.
    '''
    img_arr = load_image(size, content_path)
    img_arr = img_norm(img_arr)
    return img_arr

def load_tile_image(size, tilesize, path):
    '''Load image from disc and return the created tiled image as np array.
        Args:
            size: The desired dimensions of output image.
            tilesize: The desired dimensions of output unit tile image.
            path: The full path to the imag to load.
    '''
    width, height = size
    tile_width, tile_height = tilesize
    
    if (tile_width >  width) or (tile_height > height):
        raise ValueError(
        'Tile size needs to be smaller than the image size')
    img = Image.open(path)
    
    tile_img = img.resize((tile_width, tile_height))     
    tile_img = htile_style(img, int(width/tile_width))
    img = vtile_style(tile_img, int(height/tile_height))
    
    img = img.resize((width, height))
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, 0)
    return img_arr
    
def get_style_tile(size, tilesize, style_path):
    '''Get style image from disc and return the created tile image as np array
        Args: 
            size: The desired dimensions of output image.
            tilesize: The desired dimensions of output unit tile image.
            style_path: The full path to the imag to load.            
    '''
    style_arr = load_tile_image(size, tilesize, style_path)
    style_arr = img_norm(style_arr[:,:,:,:3])
    return style_arr


def get_style(size, style_path):
    '''Get style image from disc and return as a np array.
        Args:
            size: The desired dimensions of the output image.
            style_path: The full path of the style image to load.
    '''
    style_arr = load_image(size, style_path)
    style_arr = img_norm(style_arr[:,:,:,:3])
    return style_arr

def htile_style(style, num):
    '''Get style image and the number of tiles horizontally and return a new image object
        Args:
            style: the style image
            num: the desired number of tiles horizontally        
    '''
    imgs = [style for i in range(num)]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb

def vtile_style(style, num):
    '''Get style image and the number of tiles vertically and return a new image object
        Args:
            style: the style image
            num: the desired number of tiles vertically        
    '''
    imgs = [style for i in range(num)]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb

def deprocess(img_arr):    
    '''Returns processed image back to normal as a np array 
        Args:
            img_arr: a previously processed image np array  
    '''
    rank  = len(img_arr.shape)
    if (rank == 4):
        # Remove extra batch dimension
        img_arr = np.squeeze(img_arr, axis = 0)
        
    #flip the channels from BRG to RBG    
    img_arr = img_arr[:, :, ::-1] 
    
    # Remove zero-center by image mean pixel
    imagenet_mean = [123.68, 116.779, 103.939]
    rn_mean = np.array((imagenet_mean), dtype=np.float32)   
    img_arr = img_arr + rn_mean   
    img_arr = np.clip(img_arr, 0, 255).astype('uint8')   # Clip for better quality image
    
    return img_arr

def plot_arr(img_arr):
    '''Plot a image with a processed image array
        Args:
            arr: a previously processed image np array
    '''    
    img_arr = deprocess(img_arr)
    plt.imshow(img_arr)
    return

class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

