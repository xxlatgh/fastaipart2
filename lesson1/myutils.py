from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import importlib
import utils2; importlib.reload(utils2)
from utils2 import *
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from keras import metrics
from vgg16_avg import VGG16_Avg

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

def get_img_arr(img, rn_mean):
    preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, 0)
    img_arr = preproc(img_arr)
    return img_arr

#def get_style_arr(style):


def get_content(content_path):
    img = Image.open(content_path)
    imagenet_mean = [123.68, 116.779, 103.939]
    rn_mean = np.array((imagenet_mean), dtype=np.float32)
    img_arr = get_image_arr(img, rn_mean)
    shp = img_arr.shape
    return img_arr, shp

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

def get_content():
    content_result_path = '/home/ubuntu/data/results/cafesociety/contents'
    content_path = '/Users/xinxin/projectnotes/toddlerartist/content/dortha.jpg'
    content_model
    x = get_initial()
    x = solve_image(evaluator, iterations, x)
    Image.open(content_result_path + 'results/res_at_iteration_9.png')
    return

def get_style():
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
