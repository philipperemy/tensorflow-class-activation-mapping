import numpy as np

from class_activation_map import *
from helpers.file_logger import FileLogger
from read_caltech import read_caltech
from utils import load_image
from vgg16 import vgg16


# saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
# init_learning_rate *= 0.99
# grads_and_vars = optimizer.compute_gradients(loss_tf)
# new_grads_and_vars = []
# for gv in grads_and_vars:
#    if 'conv5_3' in gv[1].name or 'GAP' in gv[1].name:
#        print('Keeping gradient the same for {}'.format(gv[1].name))
#        new_grads_and_vars.append((gv[0], gv[1]))
#    else:
#        print('0.1 * gradient for {}'.format(gv[1].name))
#        new_grads_and_vars.append((gv[0] * 0.1, gv[1]))

# grads_and_vars = map(
#    lambda gv: (gv[0], gv[1]) if ('conv5_3' in gv[1].name or 'GAP' in gv[1].name) else (gv[0] * 0.1, gv[1]),
#    grads_and_vars)

# train_op = optimizer.apply_gradients(new_grads_and_vars)


def sanity_check(sess, vgg):
    from scipy.misc import imread, imresize
    from imagenet_classes import class_names
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    prob = sess.run(vgg.probs, feed_dict={vgg.images_placeholder: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])


def uninitialized_variables():
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars


if __name__ == '__main__':
    training_filename = FileLogger('vgg.txt', ['step', 'tr_loss'])
    weight_path = 'vgg16_weights.npz'
    model_path = './models/caltech256/'
    n_epochs = 10000
    # weight_decay_rate = 0.0005
    use_cam = False  # at the beginning we don't use Class Activation Map.
    momentum = 0.9
    batch_size = 92
    im_width = 224
    max_label_count = 2

    train_set, test_set, label_dict, n_labels = read_caltech(force_generation=True, max_label_count=max_label_count)
    print('Found {} classes in the dataset.'.format(n_labels))

    images_tf = tf.placeholder(tf.float32, [None, im_width, im_width, 3], name='images')
    labels_tf = tf.placeholder(tf.int64, [None], name='labels')

    sess = tf.Session()
    model = vgg16(images_tf, weight_path, sess, class_activation_map=use_cam, num_classes=n_labels)
    top_conv = model.pool5
    y = model.output

    if use_cam:
        class_activation_map = get_class_map(0, top_conv, im_width, gap_w=model.gap_w)
        sess.run(tf.initialize_variables([model.gap_w]))

    loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, labels_tf))

    # weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
    # weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
    # loss_tf += weight_decay

    # saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
    # step_start = restore(sess, saver, query='checkpoints/caltech*')

    trainable_variables = tf.trainable_variables()
    print('trainable_variables = [' + ',\n'.join([v.name for v in trainable_variables]) + ']')
    train_step = tf.train.AdamOptimizer().minimize(loss_tf, var_list=trainable_variables)
    sess.run(tf.initialize_variables(uninitialized_variables()))
    grad_step = 0


    def apply_model(target_set, iteration, is_training=True):
        global grad_step
        global training_filename
        print('len dataset = {}, is_training = {}'.format(len(target_set), is_training))
        loss_list = []
        acc_list = []
        for st in range(0, max(len(target_set), len(target_set) - batch_size), batch_size):
            # print(st)
            cur_data = target_set[st:st + batch_size]
            cur_img_paths = cur_data['image_path'].values
            cur_imgs = np.array(list(map(lambda x: load_image(x, im_width), cur_img_paths)))
            cur_labels = cur_data['label'].values
            feed_dict = {
                images_tf: cur_imgs,
                labels_tf: cur_labels
            }
            if is_training:
                _, loss_val, output_values = sess.run([train_step, loss_tf, y], feed_dict=feed_dict)
                grad_step += batch_size
                training_filename.write([grad_step, loss_val])
            else:
                loss_val, output_values = sess.run([loss_tf, y], feed_dict=feed_dict)
            loss_list.append(loss_val)
            predictions = output_values.argmax(axis=1)
            print('cur_labels = {}'.format(cur_labels))
            print('predictions = {}'.format(predictions))
            print('cur_labels = {}'.format(cur_labels))
            print('loss = {}'.format(loss_val))
            accuracy = np.mean(predictions == cur_labels)
            acc_list.append(accuracy)

        tag = 'tr' if is_training else 'te'
        print('Epoch = {}, loss = {}, acc = {}, target = {}'.format(iteration, np.mean(loss_list), np.mean(acc_list),
                                                                    tag))


    for epoch in range(n_epochs):
        train_set.index = list(range(len(train_set)))
        train_set = train_set.ix[np.random.permutation(len(train_set))]

        apply_model(train_set, epoch, is_training=True)
        apply_model(test_set, epoch, is_training=False)
