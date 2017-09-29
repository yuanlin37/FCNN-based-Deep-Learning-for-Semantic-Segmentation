import scipy.misc as misc
import numpy as np

filename = r'C:\dl\Data_zoo\MIT_SceneParsing\ADEChallengeData2016\annotations\training\ADE_train_00000008.png'
img1 = misc.imread(filename)
print('{}_{}', img1.shape, img1.dtype)

filename = r'C:\dl\Data_zoo\data_road\data_road\training\gt_image_2\um_lane_000007.png'
img2 = misc.imread(filename)
img2 = np.all(img2 == np.array([255, 0, 0]), axis=2)
print('{}_{}', img2.shape, img2.dtype)