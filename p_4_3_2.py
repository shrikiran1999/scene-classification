
import nbimporter
import numpy as np
import scipy.ndimage
from skimage import io
import skimage.transform
import os,time
import util
import multiprocess
import threading
import queue
import torch
import torchvision
import torchvision.transforms

def preprocess_image(image):
    '''
    Preprocesses the image to load into the prebuilt network.

    [input]
    * image: numpy.ndarray of shape (H,W,3)

    [output]
    * image_processed: torch.array of shape (3,H,W)
    '''

    # ----- TODO -----
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]
    '''
    HINTS:
    1.> Resize the image (look into skimage.transform.resize)
    2.> normalize the image
    3.> convert the image from numpy to torch
    '''
    # YOUR CODE HERE
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    image_resized = skimage.transform.resize(image, (224, 224))
    channels = image.shape[2]
    channels_normalized = []
    for c in range(channels):
        # print(image_resized[:,:,c])
        pix_norm = (image_resized[:,:,c]-mean[c])/std[c]
        # print(pix_norm)
        channels_normalized.append(pix_norm)
    
    # print(len(channels_normalized))
    # for channel in channels_normalized:
    #     channel = skimage.transform.resize(channel, (224,224))
    image_processed = np.stack((channels_normalized))
    image_processed = torch.from_numpy(image_processed)

    # raise NotImplementedError()
    return image_processed