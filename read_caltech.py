import os

import numpy as np
import pandas as pd

from utils import mkdir_p

DATASET_PATH = './data/256_ObjectCategories/'
PROCESSED_DATASET_PATH = '/tmp/caltech/'

TRAINSET_PATH = PROCESSED_DATASET_PATH + 'train.pickle'
TESTSET_PATH = PROCESSED_DATASET_PATH + 'test.pickle'
LABEL_DICT_PATH = PROCESSED_DATASET_PATH + 'label_dict.pickle'

mkdir_p(PROCESSED_DATASET_PATH)


def read_caltech(force_generation=False, max_label_count=None):
    if force_generation or not os.path.exists(TRAINSET_PATH):
        image_dir_list = os.listdir(DATASET_PATH)

        if max_label_count is not None:
            print('GOING TO TRUNCATE THE DATASET TO ONLY {} CLASSES.'.format(max_label_count))
            image_dir_list = image_dir_list[:max_label_count]

        label_pairs = map(lambda x: x.split('.'), image_dir_list)
        labels, label_names = zip(*label_pairs)
        labels = map(lambda x: int(x), labels)

        label_dict = pd.Series(labels, index=label_names)
        label_dict -= 1
        n_labels = len(label_dict)

        image_paths_per_label = list(
            map(lambda one_dir: list(map(lambda one_file: os.path.join(DATASET_PATH, one_dir, one_file),
                                         os.listdir(os.path.join(DATASET_PATH, one_dir)))), image_dir_list))
        # save just 10% for the testing set. classes have 100 images. 10 are for the testing set.
        image_paths_train = np.hstack(list(map(lambda one_class: one_class[:-10], image_paths_per_label)))
        image_paths_test = np.hstack(list(map(lambda one_class: one_class[-10:], image_paths_per_label)))

        train_set = pd.DataFrame({'image_path': image_paths_train})
        test_set = pd.DataFrame({'image_path': image_paths_test})

        # keep only the files with JPG.
        # ./data/256_ObjectCategories/001.ak47/001_0001.jpg
        # Label is 1. Label name is ak47.
        train_set = train_set[train_set['image_path'].map(lambda x: x.endswith('.jpg'))]
        train_set['label'] = train_set['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
        train_set['label_name'] = train_set['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

        # same for the testing set.
        test_set = test_set[test_set['image_path'].map(lambda x: x.endswith('.jpg'))]
        test_set['label'] = test_set['image_path'].map(lambda x: int(x.split('/')[-2].split('.')[0]) - 1)
        test_set['label_name'] = test_set['image_path'].map(lambda x: x.split('/')[-2].split('.')[1])

        label_dict.to_pickle(LABEL_DICT_PATH)
        train_set.to_pickle(TRAINSET_PATH)
        test_set.to_pickle(TESTSET_PATH)
    else:
        train_set = pd.read_pickle(TRAINSET_PATH)
        test_set = pd.read_pickle(TESTSET_PATH)
        label_dict = pd.read_pickle(LABEL_DICT_PATH)
        n_labels = len(label_dict)
    return train_set, test_set, label_dict, n_labels


if __name__ == '__main__':
    read_caltech(force_generation=True, max_label_count=2)
