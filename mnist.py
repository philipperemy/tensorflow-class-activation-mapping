from class_activation_map import *
from lenet_slim import le_net
from utils import *

batch_size = 256
dataset_percentage = 1.0 # 1.0 takes 100k rows. 0.1 takes 10k rows.

if __name__ == '__main__':
    [images_train, labels_train], [images_test, labels_test] = read_dataset(dataset_percentage)
    print('Finished reading the dataset...')

    im_width = images_train.shape[1]
    im_height = images_train.shape[1]
    assert im_height == im_width

    x = tf.placeholder(tf.float32, (None, im_width, im_width, 3))
    y, top_conv = le_net(images=x, num_classes=10)
    class_activation_map = get_class_map(0, top_conv, im_width)

    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)

    sess = tf.Session()
    sess.run(init)
    step_start = restore(sess, saver)
    print('Finished initializing the model...')

    for i in range(step_start, 100000):
        print(i)
        batch_xs, batch_ys, _ = next_batch(images_train, labels_train, i, batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            save(sess, saver, i)
            accuracy_list = []
            j = 0
            while True:
                batch_xt, batch_yt, reset = next_batch(images_test, labels_test, j, batch_size, debug=False)
                if reset:
                    break
                accuracy_list.append(sess.run(accuracy, feed_dict={x: batch_xt, y_: batch_yt}))
                j += 1
            print('steps =', i * batch_size, 'mean accuracy =', np.mean(accuracy_list))

            inspect_class_activation_map(sess, class_activation_map, top_conv, images_test,
                                         labels_test, i, 50, x, y_, y)
