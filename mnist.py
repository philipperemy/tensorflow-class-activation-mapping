import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from skimage.io import imsave
from camap import get_classmap
from lenet_slim import le_net
from utils import read_dataset
from glob import glob
from natsort import natsorted, ns

import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def next_batch(arr, arr2, index, slice_size, debug=False):
    reset = False
    index *= batch_size
    updated_index = index % len(arr)
    if updated_index + slice_size > len(arr):
        updated_index = 0
        reset = True
    beg = updated_index
    end = updated_index + slice_size
    if debug:
        print(beg, end)
    return arr[beg:end], arr2[beg:end], reset


[images_train, labels_train], [images_test, labels_test] = read_dataset(0.5)

im_width = images_train.shape[1]
im_height = images_train.shape[1]
assert im_height == im_width

# print(labels_test[-5])
# plt.imshow(images_test[-5])
# plt.show()

# mnist.train.images.shape = (55000, 784)
# mnist.train.labels.shape = (55000, 10)

x = tf.placeholder(tf.float32, (None, im_width, im_width, 3))
y, last_conv, _ = le_net(images=x, num_classes=10, is_training=True)
class_activation_map = get_classmap(0, last_conv, im_width)

y_ = tf.placeholder(tf.int64, [None])
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

sess = tf.Session()
sess.run(init)

# steps = 4300800 mean accuracy = 0.950446
# steps = 6476800 mean accuracy = 0.953348
# steps = 23475200 mean accuracy = 0.965719
batch_size = 256


def inspect_cmap(_images_test, _labels_test, i, size):
    for s in range(size):
        output_dir = 'out/{}/'.format(s)
        mkdir_p(output_dir)
        imsave('{}/image_test.png'.format(output_dir), _images_test[s])
        img = _images_test[s:s + 1]
        label = _labels_test[s:s + 1]
        conv6_val, output_val = sess.run([last_conv, y], feed_dict={x: img})
        classmap_answer = sess.run(class_activation_map, feed_dict={y_: label, last_conv: conv6_val})
        classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_answer))
        for vis, ori in zip(classmap_vis, img):
            # print(ori.shape)
            # print(vis.shape)
            plt.imshow(1 - ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            plt.savefig('{}/cmap_{}.png'.format(output_dir, i))
            plt.close()


checkpoints = natsorted(glob('checkpoints/my-model*'), key=lambda y: y.lower())
start_i = 0
if len(checkpoints) > 0:
    checkpoint = checkpoints[-2]
    saver.restore(sess, checkpoint)
    print('checkpoint restored =', checkpoint)
    start_i = int(checkpoint.split('-')[-1]) + 1

for i in range(start_i, 100000):

    batch_xs, batch_ys, _ = next_batch(images_train, labels_train, i, batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        saver.save(sess, 'checkpoints/my-model', global_step=i)
        accuracy_list = []
        j = 0
        while True:
            batch_xt, batch_yt, reset = next_batch(images_test, labels_test, j, batch_size, debug=False)
            if reset:
                break
            accuracy_list.append(sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt}))
            j += 1
        print(accuracy_list)
        print('steps =', i * batch_size, 'mean accuracy =', np.mean(accuracy_list))

        inspect_cmap(images_test, labels_test, i, size=10)
