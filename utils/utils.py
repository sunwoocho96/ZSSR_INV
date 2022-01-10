import os
import time
import torch
import math
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from torch.nn import functional as F
from scipy.ndimage import measurements, interpolation
from utils.imresize import imresize


def preprocess_kernels(kernels, conf):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if kernels is not None:
        return [kernel_shift(loadmat(kernel)['Kernel'], sf)
                for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)

def father_to_son(hr_father, kernel, sf):
    # Create son out of the father by downscaling and if indicated adding noise
    lr_son = imresize(hr_father, 1.0 / sf, kernel=kernel)
    return np.clip(lr_son, 0, 1) 


def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    im = np.double(im) / 255.
    return im

def write_image(img, path):
    print('[*] img in write image', img.shape)
    print('[*] img squeezed ', img.dtype)
    if np.max(img) <= 1 :
        img = np.uint8(np.clip(np.round(img*255.), 0., 255.))
    im = Image.fromarray(img)
    im.save(path)

def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)

def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / \
                                                                               prob_map.flatten().shape[0]
    return prob_vec

def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()

def im2tensor(im_np):
    """Copy the image to the gpu & converts to range [-1,1]"""
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1)) * 2.0 - 1.0).unsqueeze(0).cuda()

def tensor2im(im_t):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0)) + 1) / 2.0 * 255.0), 0, 255)
    return im_np.astype(np.uint8)    

def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im

    
def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded

    
def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min