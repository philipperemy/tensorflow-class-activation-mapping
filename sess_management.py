import tensorflow as tf


if __name__ == '__main__':

    with tf.Session() as session:

        x = tf.Variable([42.0, 42.1, 42.3], name='x')
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
        x = tf.add(x, x)

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, 'checkpoints/model.ckpt')
            print('model loaded.')
        else:
            print('no checkpoint found.')
        ckpt = tf.train.get_checkpoint_state('checkpoints')
        saver.save(session, 'checkpoints/model.ckpt')
        session.run(tf.initialize_all_variables())
        print(session.run(x))
