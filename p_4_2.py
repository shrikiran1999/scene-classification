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


def get_image_feature(args):
    '''
    Extracts deep features from the prebuilt VGG-16 network.
    This is a function run by a subprocess.
    [input]
    * i: index of training image
    * image_path: path of image file
    * vgg16: prebuilt VGG-16 network.
    
    [output]
    * feat: evaluated deep feature
    '''
    i, image_path, vgg16 = args
    image = io.imread(image_path) / 255
    
    '''
    HINTS:
    1.> Think along the lines of evaluate_deep_extractor
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    vgg16_weights = util.get_VGG16_weights()
    img_torch = preprocess_image(image)
    # print(img_torch.shape)
    # print(np.transpose(img_torch.numpy(), (1,2,0)).shape)
    # feat = extract_deep_feature(np.transpose(img_torch.numpy(), (1,2,0)), vgg16_weights)
    
    with torch.no_grad():
        vgg_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-3])
        vgg_feat_feat = vgg16.features(img_torch[None, ])
        vgg_feat_feat = vgg_classifier(vgg_feat_feat.flatten())
    
    # return np.sum(np.abs(vgg_feat_feat.numpy() - feat))
    # raise NotImplementedError()
    feat = vgg_feat_feat.numpy()
    return [i,feat]


