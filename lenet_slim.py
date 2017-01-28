import tensorflow as tf

slim = tf.contrib.slim


def le_net(images, num_classes=10, is_training=False,
           dropout_keep_prob=0.5,
           prediction_fn=slim.softmax,
           scope='LeNet'):
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        # net = slim.flatten(net)
        # end_points['Flatten'] = net

        gap = tf.reduce_mean(net, (1, 2))
        with tf.variable_scope('GAP'):
            gap_w = tf.get_variable(
                'W',
                shape=[64, 10],
                initializer=tf.random_normal_initializer(0., 0.01))
        logits = tf.matmul(gap, gap_w)

        # net = slim.fully_connected(net, 1024, scope='fc3')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                   scope='dropout3')
        # logits = slim.fully_connected(net, num_classes, activation_fn=None,
        #                              scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, net, end_points


# le_net.default_image_size = 28


def lenet_arg_scope(weight_decay=0.0):
    """Defines the default lenet argument scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc
