import errno
import os
from glob import glob

import numpy as np
import skimage.io
import skimage.transform
from natsort import natsorted


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def new_conv_layer(bottom, filter_shape, name):
    import tensorflow as tf
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


def load_image(path, im_width=None):
    try:
        img = skimage.io.imread(path).astype(float)
    except:
        return None

    if img is None:
        return None
    if len(img.shape) < 2:
        return None
    if len(img.shape) == 4:
        return None
    if len(img.shape) == 2:
        img = np.tile(img[:, :, None], 3)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] > 4:
        return None

    img /= 255.

    if im_width is not None:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
        resized_img = skimage.transform.resize(crop_img, [im_width, im_width])
        return resized_img
    else:
        return img


def next_batch(arr, arr2, index, slice_size, debug=False):
    from mnist import batch_size
    has_reset = False
    index *= batch_size
    updated_index = index % len(arr)
    if updated_index + slice_size > len(arr):
        updated_index = 0
        has_reset = True
    beg = updated_index
    end = updated_index + slice_size
    if debug:
        print(beg, end)
    return arr[beg:end], arr2[beg:end], has_reset


def restore(sess, saver, query='checkpoints/mnist-cluttered*'):
    checkpoints = natsorted(glob(query), key=lambda y: y.lower())
    start_i = 0
    if len(checkpoints) > 0:
        checkpoint = checkpoints[-2]
        saver.restore(sess, checkpoint)
        print('checkpoint restored =', checkpoint)
        start_i = int(checkpoint.split('-')[-1]) + 1
    return start_i


def save(sess, saver, i, query='checkpoints/mnist-cluttered'):
    mkdir_p('checkpoints')
    saver.save(sess, query, global_step=i)


if __name__ == '__main__':
    read_dataset(0.01)
