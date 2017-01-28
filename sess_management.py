import tensorflow as tf


if __name__ == '__main__':

    # graph
    x = tf.Variable(12, name='x')
    x = tf.add(x, x)

    saver = tf.train.Saver(tf.all_variables())

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    #for step in range(10):
    #    x_val = session.run(x)
    #    print(x_val)
    #    saver.save(session, 'checkpoints/my-model', global_step=step)

    print(saver.last_checkpoints)
    saver.restore(session, 'checkpoints/my-model-9')
    print(session.run(x))
