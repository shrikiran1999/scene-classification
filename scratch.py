bow = np.load('dictionary.npy')
# path_img = "./data/aquarium/sun_aztvjgubyrgvirup.jpg"
path_img = "./data/desert/sun_aaqyzvrweabdxjzo.jpg"
image = io.imread(path_img)
image = image.astype('float')/255
word_map = get_visual_words(image, bow)
print(word_map[0])
print(word_map.shape)

from util import save_wordmap
save_wordmap(word_map, 'aquarium')


from p1_1 import extract_filter_responses
from p1_1 import get_harris_corners
from p1_1 import compute_dictionary_one_image

before compute_dictionary


#  2.1
#  plotting histogram, need to extend to 5 images, copy list
from skimage import io
bow = np.load('dictionary.npy')
# path_img = "./data/aquarium/sun_aztvjgubyrgvirup.jpg"
path_img = "./data/aquarium/sun_aydaknxraiwghvmi.jpg"
image = io.imread(path_img)
image = image.astype('float')/255
word_map = get_visual_words(image, bow)
###### add bins to return statement in fn during plotting ######
hist, bins = get_feature_from_wordmap(word_map, len(bow)) 
print(len(bins))
print(bins)
print(np.sum(hist * np.diff(bins)))
# print(word_map[0])
# print(word_map.shape)

# from util import save_wordmap
# save_wordmap(word_map, 'Q1_1_2')

# import matplotlib.pyplot as plt
# plt.hist(word_map, bins, density=True)
# plt.show()

new_patch_width = (2**l)*(patch_width)
new_patch_height = (2**l)*(patch_height)

q2.2

from skimage import io
bow = np.load('dictionary.npy')
path_img = "./data/aquarium/sun_aairflxfskjrkepm.jpg"
image = io.imread(path_img)
image = image.astype('float')/255
word_map = get_visual_words(image, bow)
hist_all = get_feature_from_wordmap_SPM(word_map, 3, 200)



q3

# NOTE: comment out the lines below before submitting to gradescope
conf_matrix, accuracy = evaluate_recognition_system()
# We expect the accuracy to be greater than 0.45
print("Accuracy:", accuracy)