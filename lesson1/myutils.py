from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import PIL 

import importlib
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


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
    width, height = size
    tile_width, tile_height = tilesize
    img = Image.open(path)
    
    tile_img = img.resize((tile_width, tile_height))     
    tile_img = htile_style(img, int(width/tile_width))
    img = vtile_style(tile_img, int(height/tile_height))
    
    img = img.resize((width, height))
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, 0)
    return img_arr
    
def get_style_tile(size, tilesize, style_path):
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

def solve_image(eval_obj, niter, x):
    deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(eval_obj.loss, x.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        x = np.clip(x, -127,127)
        print('Current loss value:', min_val)
        imsave(f'{result_path}/res_at_iteration_{i}.png', deproc(x.copy(), shp)[0])
    return x

def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation 
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()

def style_loss(x, targ): 
    return metrics.mse(gram_matrix(x), gram_matrix(targ))

def content_model(img_arr):
    model = VGG16_Avg(include_top = False)
    layer = model.get_layer('block3_conv2').output
    layer_model = Model(model.input, layer)
    target = K.variable(layer_model.predict(img_arr))
    loss = metrics.mse(layer, targ)
    grads = K.gradients(loss, model.input)
    fn = K.function([model.input], [loss]+grads)
    evaluator = Evaluator(fn)
    return evaluator

def get_initial():
    rand_img = lambda shape: np.random.uniform(-2.5, 2.5, shape)/100
    x = rand_img(shp)
#    plt.imshow(x[0])
    return x

def get_content_model():
    content_result_path = '/home/ubuntu/data/results/cafesociety/contents'
    content_path = '/Users/xinxin/projectnotes/toddlerartist/content/dortha.jpg'
    content_model
    x = get_initial()
    x = solve_image(evaluator, iterations, x)
    Image.open(content_result_path + 'results/res_at_iteration_9.png')
    return

def get_style_model():
    style_result_path = '/home/ubuntu/data/results/cafesociety/styles'
    style_path = '/Users/xinxin/projectnotes/toddlerartist/style/flowerplat.jpg'
    x = get_initial()
    x = solve_image(evaluator, iterations, x)
    style_model()#define style_model
    Image.open(style_result_path + 'results/res_at_iteration_9.png')
    return

def get_neural_style():
    merge_result_path = '/home/ubuntu/data/results/miaavery/'
    #tobecontinue

    

def img_rotate_expand(img, crop_dim):
    newimg = img.rotate(90, expand=1).crop(crop_dim)
    return newimg

def img_smaller(img, factor):
    newimg = img.resize(np.divide(img.size, factor).astype('int32'))
    return newimg

def define_crop_dim(left, upper, right, lower):
    crop_dim = [left, upper, right, lower]
    return crop_dim

def image_crop(img, crop_dim):
    newimg = img.crop(crop_dim)
    return newimg

def plots(ims, merge_path, figsize = (12, 6), row=1, cols=None, interp=None, titles=None, cmap=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        subplot = f.add_subplot(rows, cols, i+1)
        if titles is not None:
            subplot.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=interp, cmap=cmap)
        plt.axis('off')# try chagne it 'on'
        plt.subplots_adjust(hspace = 0.100)

    f.savefig(merge_path+'/summary.png') #need to test it
    return
    
'''
def main():
    content_path = ''
    style_path = ''
    get_img_arr(img=Image.open(content_path))
    get_style_arr(style=Image.open(style_path))

    get_content()
    get_style()
    get_neural_style()

if __name__ == '__main__':
    main()
'''
