import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import PIL
import sys
import torch

from PIL import Image, ImageFilter, ImageEnhance
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from torch import nn
from torchvision import datasets, transforms, models

# -- General utility --
class Bunch(object):
    def __init__(self, param_dict={}):
        self.__dict__.update(param_dict)
    def get(self, param_name, default_ret=None):
        return self.__dict__.get(param_name, default_ret)
    
def to_numpy(array):
    #if type(array) is torch.Tensor:
    try:
        return array.detach().cpu().numpy()
    except:
        return np.array(array)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def add_outside_module(path):
    if path not in sys.path:
        sys.path.append(path)


# -- For loading images --
def load_image(path):
    image = cv2.imread(str(path))
    assert image is not None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_width = (image.shape[1] // 32) * 32
    # image_height = (image.shape[0] // 32) * 32
    # image = image[:image_height, :image_width]
    return image
def load_images(args):
    if not os.path.exists(args.path_to_images):
        return
    if os.path.isdir(args.path_to_images):
        for p in os.listdir(args.path_to_images):
            p = os.path.join(args.path_to_images, p)
            yield load_image(p), p
    else:
        yield load_image(args.path_to_images), args.path_to_images

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

# -- For plotting / displaying images --
def display_tensor_im(x, flip_color=False):
    if type(x) is PIL.Image.Image:
        x = np.array(x)
    if type(x) is np.ndarray:
        x = torch.Tensor(x)
    if len(x.shape) == 2:
        x = x.unsqueeze(-1)
        x = tile(x.detach().cpu(), dim=-1, n_tile=3)
    if np.argmin(x.shape) == 0:
        x = x.permute(1,2,0)
    if flip_color:
        # change from BGR to RGB
        x = x[:, :, [2,1,0]]
    if torch.max(x) > 1:
        x = x/255.
    return x

def display_image(x1, flip_color1=False, flip_color2=False):
    plt.figure(figsize=(12,8))
    im1 = display_tensor_im(x1, flip_color1)
    plt.imshow(im1)
    plt.show()

def gen_image_pair(x1,x2, flip_color1=False, flip_color2=False):
    im1 = display_tensor_im(x1, flip_color1)
    im2 = display_tensor_im(x2, flip_color2)
    if im1.shape[0] > im2.shape[0]:
        im2 = F.pad(input=im2, pad=(0, 0, 0, 0, 0, im1.shape[0]-im2.shape[0]), mode='constant', value=0)
    elif im2.shape[0] > im1.shape[0]:
        im1 = F.pad(input=im1, pad=(0, 0, 0, 0, 0, im2.shape[0]-im1.shape[0]), mode='constant', value=0)
    return torch.cat((im1, im2), dim=1)
def display_image_pair(x1,x2, flip_color1=False, flip_color2=False):
    plt.figure(figsize=(12,16))
    plt.imshow(gen_image_pair(x1, x2, flip_color1=flip_color1, flip_color2=flip_color2))
    plt.show()

def display_image_and_mask(image, mask_image, flip_color2=True):
    display_image_pair(image, mask_image, flip_color1=False, flip_color2=flip_color2)

def display_image_and_mask_and_saliency(image, mask_image, saliency_im):
    x1 = gen_image_pair(image, mask_image, flip_color1=False, flip_color2=False)
    display_image_pair(x1, saliency_im, flip_color1=False, flip_color2=False)


# -- Processing and analysis of images --
def gen_patches_from_im(im, mask, patch_size=25, max_patches=1000):
    p = extract_patches_2d(torch.cat((im, mask), axis=-1), 
                           (patch_size, patch_size), 
                           max_patches=max_patches)
    skin_overlap = [np.sum(pi[:,:,-1]>0) / np.size(pi[:,:,-1]) for pi in p]
    idxs = np.argsort(skin_overlap)[::-1]
    filtered_p = [p[idx][:,:,:-1] for idx in idxs if skin_overlap[idx] == 1]
    if len(filtered_p) < max_patches // 10:
        filtered_p = [p[idxs[i]][:,:,:-1] for i in range(max_patches // 10) if skin_overlap[idxs[i]] > 0.9]
    if len(filtered_p) < 1:
        return None
    return (255*np.stack(filtered_p)).astype(np.uint8)

def gen_center_mask(image, ratio=0.5):
    mask = torch.zeros(image.shape[:2])
    im_w, im_h = mask.shape
    s1, s2 = (0.5-ratio/2), (0.5+ratio/2)
    mask[int(im_w*s1):int(im_w*s2), int(im_h*s1):int(im_h*s2)] = 1
    return mask.astype(bool)
def apply_center_mask(image, ratio=0.5):
    return image*gen_center_mask(image, ratio)[:,:,None]
def apply_center_crop(image, ratio=0.5):
    im_w, im_h = image.shape[:2]
    s1, s2 = (0.5-ratio/2), (0.5+ratio/2)
    return image[int(im_w*s1):int(im_w*s2), int(im_h*s1):int(im_h*s2)]



# -- For image segmentation --
def pil_to_skin_mask(img, t):
    """Source: https://github.com/CHEREF-Mehdi/SkinDetection/blob/master/SkinDetection.py"""
    
    img = img.astype(np.uint8)
    
    #skin color range for hsv color space 
    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 15-3*t, 0), (17+(238/5)*t,170+17*t,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #skin color range for YCbCr color space 
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135-27*t, 85-17*t), (255,180+15*t,135+23*t)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    
    #merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask = cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    global_result = (global_mask.astype(bool))
    
    return global_result


def _featurize_rgb(x): 
    return np.concatenate((cv2.cvtColor(x[None,:,:], cv2.COLOR_RGB2YCrCb)[0],
                           cv2.cvtColor(x[None,:,:], cv2.COLOR_RGB2HSV)[0]
                         ), axis=-1)
def _inv_featurize_rgb(x):
    return np.mean((cv2.cvtColor(x[None,:,:3], cv2.COLOR_YCrCb2RGB),
                    cv2.cvtColor(x[None,:,3:], cv2.COLOR_HSV2RGB)
                 ), axis=0)
class SkinDistribution:
    def __init__(self, model_path, data_path="data/skin_data/Skin_NonSkin.txt"):
        if not os.path.exists(model_path):
            with open(data_path, 'r') as f:
                bgrl = f.readlines()
            RGB = np.array([[int(i) for i in bgrli.split()[:3][::-1]] for bgrli in bgrl]).astype(np.uint8)
            l = np.array([int(bgrli.split()[3]) for bgrli in bgrl])

            
            preprocess = FunctionTransformer(func=_featurize_rgb, inverse_func=_inv_featurize_rgb)
            pca = PCA(n_components=6)
            gmm = GaussianMixture(n_components=3)
            pipe = Pipeline(steps=[('color_space', preprocess), ('pca', pca), ('gmm', gmm)])

            pipe.fit(RGB[np.where(l==1)])

            with open("models/skin_dist.pkl", 'wb') as f:
                pickle.dump(pipe, f)

        with open(model_path, 'rb') as f:
            pipe = pickle.load(f)
        self.model = pipe.steps[-1][1]
        self.preprocess = pipe
        self.preprocess.steps = self.preprocess.steps[:-1]
    def __call__(self, image):
        scores = self.model.score_samples(self.preprocess.transform(image.reshape((-1,3))))
        return -scores.reshape(image.shape[:2]) / 10.655672255998063
