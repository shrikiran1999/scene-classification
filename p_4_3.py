
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
    
def evaluate_test_image(args):
    # YOUR CODE HERE
    i, image_path, vgg16, trained_features, train_labels = args
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
    N = trained_features.shape[0]
    feat = vgg_feat_feat.numpy() #(K,)
    feat_repeated = np.stack(([feat]*N)) # (N, K)
    distances = np.linalg.norm((feat_repeated, trained_features), axis=1)
    idx_min = np.argmin(distances)
    print(distances.shape)
    # raise NotImplementedError()
    pred_label = train_labels[idx_min]
    return [i, pred_label]