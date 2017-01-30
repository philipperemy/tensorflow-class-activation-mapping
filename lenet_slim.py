import tensorflow as tf

slim = tf.contrib.slim


def le_net(images, num_classes=10, scope='LeNet'):
    with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        gap = tf.reduce_mean(net, (1, 2))
        with tf.variable_scope('GAP'):
            gap_w = tf.get_variable('W', shape=[64, 10], initializer=tf.random_normal_initializer(0., 0.01))
        logits = tf.matmul(gap, gap_w)
    return logits, net


def le_net_arg_scope(weight_decay=0.0):
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc
