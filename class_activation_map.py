import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.io import imsave

from utils import mkdir_p


def get_class_map(label, conv, im_width):
    output_channels = int(conv.get_shape()[-1])
    conv_resized = tf.image.resize_bilinear(conv, [im_width, im_width])
    with tf.variable_scope('LeNet/GAP', reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable('W')), label)
        label_w = tf.reshape(label_w, [-1, output_channels, 1])
    conv_resized = tf.reshape(conv_resized, [-1, im_width * im_width, output_channels])
    classmap = tf.batch_matmul(conv_resized, label_w)
    classmap = tf.reshape(classmap, [-1, im_width, im_width])
    return classmap


def inspect_class_activation_map(sess, class_activation_map, top_conv,
                                 images_test, labels_test, global_step,
                                 num_images, x, y_, y):
    for s in range(num_images):
        output_dir = 'out/img_{}/'.format(s)
        mkdir_p(output_dir)
        imsave('{}/image_test.png'.format(output_dir), images_test[s])
        img = images_test[s:s + 1]
        label = labels_test[s:s + 1]
        conv6_val, output_val = sess.run([top_conv, y], feed_dict={x: img})
        classmap_answer = sess.run(class_activation_map, feed_dict={y_: label, top_conv: conv6_val})
        classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), classmap_answer))
        for vis, ori in zip(classmap_vis, img):
            plt.imshow(1 - ori)
            plt.imshow(vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            cmap_file = '{}/cmap_{}.png'.format(output_dir, global_step)
            plt.savefig(cmap_file)
            plt.close()
