from glob import glob

import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf


def new_conv_layer(bottom, filter_shape, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable(
            "W",
            shape=filter_shape,
            initializer=tf.random_normal_initializer(0., 0.01))
        b = tf.get_variable(
            "b",
            shape=filter_shape[-1],
            initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(bottom, w, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, b)
    return bias


def read_dataset(percentage=1.0, cutoff=0.7):
    labels = []
    with open('data/labels.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            [_, label] = l.strip().split('\t')
            labels.append(int(label))
    labels = np.array(labels)
    images = []
    n = len(glob('/tmp/img/*.png'))
    assert n == len(labels)
    max_images = int(percentage * n)
    labels = labels[:max_images]
    print('found {} images.'.format(n))
    for i in range(1, n + 1):
        f = '/tmp/img/img_{}.png'.format(i)
        images.append(load_image(f))
        if i % 1000 == 0:
            print('read {} images.'.format(i))
        if len(images) == max_images:
            break
    images = np.array(images)
    print(images.shape)
    assert max_images == len(images)
    assert max_images == len(labels)

    separator = int(max_images * cutoff)
    return [images[:separator], labels[:separator]], [images[separator:], labels[separator:]]


def load_image(path):
    try:
        img = skimage.io.imread(path).astype(float)
    except:
        return None

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4: img = img[:, :, :3]
    if img.shape[2] > 4: return None

    img /= 255.

    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resized_img = skimage.transform.resize(crop_img, [224, 224])
    return img


if __name__ == '__main__':
    read_dataset(0.01)
