

# Do Not Modify
import nbimporter
from util import display_filter_responses
import numpy as np
import multiprocess
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import random
import cv2

from skimage import io
#-------------------------------------------------------------------------


def plot_harris_points(image,points):
    fig = plt.figure(1)
    for x,y in zip(points[0],points[1]):
        plt.plot(y,x,marker='v')
    plt.imshow(image)
    plt.show()

    
def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''
    
    
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(len(image.shape) == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]

    image = skimage.color.rgb2lab(image)

    filter_responses = []
    '''
    HINTS: 
    1.> Iterate through the scales (5) which can be 1, 2, 4, 8, 8$\sqrt{2}$
    2.> use scipy.ndimage.gaussian_* to create filters
    3.> Iterate over each of the three channels independently
    4.> stack the filters together to (H, W,3F) dim
    '''
    # ----- TODO -----
    
    # YOUR CODE HERE
    m, n, channels = image.shape
    assert channels==3
    scales = [1, 2, 4, 8, 8*(2**0.5)]
    
    for s in scales:
        for c in range(channels):
            if len(filter_responses)==0:
                filter_responses = np.expand_dims(np.array(scipy.ndimage.gaussian_filter(image[:, :, c], sigma=s)), axis=2)
            else:
                filter_responses = np.concatenate((filter_responses, np.expand_dims(scipy.ndimage.gaussian_filter(image[:, :, c], sigma=s), axis=2)), axis=-1)
            filter_responses = np.concatenate((filter_responses, np.expand_dims(scipy.ndimage.gaussian_laplace(image[:, :, c], sigma=s), axis=2)), axis=-1)
            filter_responses = np.concatenate((filter_responses, np.expand_dims(scipy.ndimage.gaussian_filter(image[:, :, c], sigma=s, order=(0,1)), axis=2)), axis=-1)
            filter_responses = np.concatenate((filter_responses, np.expand_dims(scipy.ndimage.gaussian_filter(image[:, :, c], sigma=s, order=(1,0)), axis=2)), axis=-1)

    assert filter_responses.shape==(m,n,60)



    # raise NotImplementedError()
    return filter_responses

def get_harris_corners(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (2, alpha) that contains interest points
    '''
    
    '''
    HINTS:
    (1) Visualize and Compare results with cv2.cornerHarris() for debug (DO NOT SUBMIT cv2's implementation)
    '''
    # ----- TODO -----
    
    ######### Actual Harris #########
    from skimage.color import rgb2gray
    from scipy import ndimage


    bw_img = rgb2gray(image)
    '''
    HINTS:
    1.> For derivative images we can use cv2.Sobel filter of 3x3 kernel size
    2.> Multiply the derivatives to get Ix * Ix, Ix * Iy, etc.
    '''
    # YOUR CODE HERE

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    Ix = cv2.Sobel(bw_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.Sobel(bw_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    Ixx = Ix*Ix
    Ixy = Ix*Iy
    Iyy = Iy*Iy
    # print(image.shape)
    # print(Ix.shape)
    sum_filter = np.ones((3, 3))
    # print(sum_filter)
    # print(Ixx.shape)
    Ixx_sum =ndimage.convolve(Ixx, sum_filter)
    Ixy_sum =ndimage.convolve(Ixy, sum_filter)
    Iyy_sum =ndimage.convolve(Iyy, sum_filter)
    R = np.zeros_like(bw_img)
    for m in range(bw_img.shape[0]):
        for n in range(bw_img.shape[1]):
            H = np.array([[Ixx_sum[m,n], Ixy_sum[m,n]],
                        [Ixy_sum[m,n], Iyy_sum[m,n]]])
            R[m,n] = np.linalg.det(H)-k*(np.trace(H)**2)
    
    interest_points_flattened = np.argsort(R.flatten())[-alpha:]
    points_of_interest = np.unravel_index(interest_points_flattened, (bw_img.shape[0], bw_img.shape[1]))
    point_coordinates = zip(points_of_interest[0], points_of_interest[1]) # (alpha, 2)

    '''
    HINTS:
    1.> Think of R = det - trace * k
    2.> We can use ndimage.convolve
    3.> sort (argsort) the values and pick the alpha larges ones
    3.> points_of_interest should have this structure [[x1,x2,x3...],[y1,y2,y3...]] (2,alpha)
        where x_i is across H and y_i is across W
    '''
    # YOUR CODE HERE
    # raise NotImplementedError()
    
    ######### Actual Harris #########
    return points_of_interest

def compute_dictionary_one_image(args):
    '''
    Extracts samples of the dictionary entries from an image. Use the the 
    harris corner detector implmented from previous question to extract 
    the point of interests. This should be a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of samples
    * image_path: path of image file

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha, 3F)
    '''
    i, alpha, image_path = args
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    f_name = 'tmp/%05d.npy' % i
    
    # ----- TODO -----
    '''
    HINTS:
    1.> Create a tmp dir to store intermediate results.
    2.> Read the image from image_path using skimage
    3.> extract filter responses and points of interest
    4.> store the response of filters at points of interest 
    '''
    # YOUR CODE HERE
    image = io.imread(image_path)
    image = image.astype('float')/255
    filter_responses = extract_filter_responses(image) # (H,W,3F)
    harris_points = get_harris_corners(image,alpha) # (2, alpha)
    point_coordinates = zip(harris_points[0], harris_points[1]) # (alpha, 2)
    sampled_features = np.array([filter_responses[x,y] for (x,y) in point_coordinates]) # (alpha, 3F)

    np.save(f_name, sampled_features)
    
    # raise NotImplementedError()

