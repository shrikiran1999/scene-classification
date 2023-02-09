import nbimporter
import numpy as np
import skimage
import multiprocess
import threading
import queue
import os,time
import math
# from ipynb.fs.defs.p1 import get_visual_words
from p1 import get_visual_words

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K (which is 200)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    '''
    HINTS:
    (1) We can use np.histogram with flattened wordmap
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    hist, bins = np.histogram(wordmap, bins=dict_size, range=(1, dict_size+1), density=True) #image ids start from 1 according to TAs; bins: (201,)
    # raise NotImplementedError()
    return hist # (200,)

def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    '''
    HINTS:
    (1) Take care of Weights 
    (2) Try to build the pyramid in Bottom Up Manner
    (3) the output array should first contain the histogram for Level 0 (top most level) , followed by Level 1, and then Level 2.
    '''
    # ----- TODO -----
    h, w = wordmap.shape
    L = layer_num - 1
    patch_width = math.floor(w / (2**L))
    patch_height = math.floor(h / (2**L))
    
    '''
    HINTS:
    1.> create an array of size (dict_size, (4**(L + 1) -1)/3) )
    2.> pre-compute the starts, ends and weights for the SPM layers L 
    '''
    # YOUR CODE HERE
    hist_all_temp = np.zeros((dict_size, int((4**(L + 1) -1)/3))) # (200, 21)
    hist, bins = np.histogram(wordmap, bins=dict_size, range=(1, dict_size+1), density=True)
    weights = [2**(l-L-1) for l in range(L, 1, -1)] # (L-1,)
    weights.append(2**(-L)) # (L,)
    weights.append(2**(-L)) # (L+1,)
    # raise NotImplementedError()
    '''
    HINTS:
    1.> Loop over the layers from L to 0
    2.> Handle the base case (Layer L) separately and then build over that
    3.> Normalize each histogram separately and also normalize the final histogram
    '''
    # YOUR CODE HERE
    # layers_hist_normalized = [] # (L+1,K*(4**L)) {order: L->0} induvidual layers to be weighted later, each element is a normalized histogram from one layer
    layers_hist_normalized_weighted = [] # (L+1,K*(4**L)) {order: L->0}
    patches_fine = []
    # patch_hists_fine = []
    for l in range(L, -1, -1):
        if l==L:
            n_patches = 4**L
            patch_hists_fine = []
             # after loop ends: (4**L, K)
            for y in range(0, int(h-h%(2**L)), patch_height):
                for x in range(0, int(w-w%(2**L)), patch_width):
                    patch = wordmap[y:y+patch_height, x:x+patch_width]
                    patches_fine.append(patch) # appending original patch
                    patch_hist = get_feature_from_wordmap(patch, dict_size) # (K,)
                    patch_hists_fine.append(patch_hist)


            layer_hist_normalized_weighted = np.concatenate((patch_hists_fine))*(weights[L-l]/n_patches)
            layers_hist_normalized_weighted.append(layer_hist_normalized_weighted) # dim of array inside append: K*(4**L)
        else:
            if l==0:
                patch_hist = get_feature_from_wordmap(wordmap, dict_size)
                layer_hist_normalized_weighted = patch_hist*weights[L-l]
                layers_hist_normalized_weighted.append(layer_hist_normalized_weighted)

            else:
                new_patches = []
                n_patches_new = 4**l
                fine_patch_idxs = np.array([i for i in range(4**(l+1))])
                fine_patch_idxs = np.reshape(fine_patch_idxs, (int((4**(l+1))**0.5), int((4**(l+1))**0.5)))
                # print(fine_patch_idxs)
                # creating new patches from the next layer's patches; irrespective of layer, 4 fine patches have to be combined to create new patch
                for i in range(0, fine_patch_idxs.shape[0], 2):
                    for j in range(0, fine_patch_idxs.shape[1], 2):
                        sub_patch_row1 = np.hstack((patches_fine[fine_patch_idxs[i][j]], patches_fine[fine_patch_idxs[i][j+1]]))
                        sub_patch_row2 = np.hstack((patches_fine[fine_patch_idxs[i+1][j]], patches_fine[fine_patch_idxs[i+1][j+1]]))
                        new_patches.append(np.vstack((sub_patch_row1, sub_patch_row2)))
                patches_fine = new_patches
                    
                patch_hists_fine = []
                for p in range(n_patches_new):
                    patch_hist = get_feature_from_wordmap(new_patches[p], dict_size)
                    patch_hists_fine.append(patch_hist)

                layer_hist_normalized_weighted = np.concatenate((patch_hists_fine))*(weights[L-l]/n_patches_new)
                layers_hist_normalized_weighted.append(layer_hist_normalized_weighted)

                                    
    layers_hist_normalized_weighted_ordered = layers_hist_normalized_weighted[::-1]
    hist_all = np.concatenate((layers_hist_normalized_weighted_ordered))
            
    # raise NotImplementedError()
    return hist_all

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    '''
    HINTS:
    (1) Consider A = [0.1,0.4,0.5] and B = [[0.2,0.3,0.5],[0.8,0.1,0.1]] then \
        similarity between element A and set B could be represented as [[0.1,0.3,0.5],[0.1,0.1,0.1]]   
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    # raise NotImplementedError()
    N = histograms.shape[0]
    word_hist_repeated = np.stack(([word_hist]*N)) # (N,K)
    min_hist_counts = np.minimum(word_hist_repeated, histograms)
    sim = np.sum(min_hist_counts, axis=1)
    
    return sim