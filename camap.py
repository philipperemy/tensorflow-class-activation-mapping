import tensorflow as tf


def get_classmap(label, conv, im_width):
    conv_resized = tf.image.resize_bilinear(conv, [im_width, im_width])
    with tf.variable_scope('LeNet/GAP', reuse=True):
        label_w = tf.gather(tf.transpose(tf.get_variable('W')), label)
        label_w = tf.reshape(label_w, [-1, 64, 1])  # [batch_size, OUTPUT_CHANNELS, 1]

    conv_resized = tf.reshape(conv_resized,
                              [-1, im_width * im_width, 64])  # [batch_size, im_width**2, OUTPUT_CHANNELS]

    classmap = tf.batch_matmul(conv_resized, label_w)
    classmap = tf.reshape(classmap, [-1, im_width, im_width])
    return classmap
