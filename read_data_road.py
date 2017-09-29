__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://kitti.is.tue.mpg.de/kitti/data_road.zip'


def read_dataset(data_dir):
    pickle_filename = "data_road.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    type = ['um_', 'umm_']
    directories = ['training', 'validation']
    image_list = {}

    for idx in np.arange(len(directories)):
        file_list = []
        image_list[directories[idx]] = []
        file_glob = os.path.join(image_dir, 'training', 'gt_image_2',  type[idx] + '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                annotation_file = f
                filename = os.path.basename(f)[:-4]
                imagename = os.path.join(image_dir, 'training', 'image_2', filename.replace('_road', '').replace('_lane', '') + '.png')
                if os.path.exists(imagename):
                    record = {'image': imagename, 'annotation': annotation_file, 'filename': filename}
                    image_list[directories[idx]].append(record)
                else:
                    print("Annotation file %s not found for %s - Skipping" % (annotation_file, filename))
                print("Annotation file %s for %s" % (annotation_file, filename))

        random.shuffle(image_list[directories[idx]])
        no_of_images = len(image_list[directories[idx]])
        print ('No. of %s files: %d' % (directories[idx], no_of_images))

    image_list['NUM_OF_CLASSESS'] = 151
    image_list['IMAGE_SIZE'] = 224
    return image_list
