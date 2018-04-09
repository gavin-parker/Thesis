from params import FLAGS
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import glob
import os

"""Train the model with the settings provided in FLAGS"""


def train(model=None, sess=None, name=time.strftime("%H:%M:%S")):
    assert model

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    if sess is None:
        sess = tf.Session(config=config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    options, run_metadata = None, None
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    if FLAGS.profile:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    tf.train.start_queue_runners(sess=sess)
    train_writer = tf.summary.FileWriter(
        "{}/{}".format(FLAGS.log_dir, name),
        sess.graph)

    saver = tf.train.Saver()
    if FLAGS.fine_tune:
        saver.restore(sess, FLAGS.test_model_dir)
    epoch_size = len(glob.glob("{}/left/*.png".format(FLAGS.train_dir)))
    epoch_size /= FLAGS.batch_size
    validation_size = len(glob.glob("{}/left/*.png".format(FLAGS.val_dir)))
    validation_size /= FLAGS.batch_size
    print("beginning training with learning rate: {}".format(FLAGS.learning_rate))
    print("Epoch size: {}".format(epoch_size))
    print("Batch size: {}".format(FLAGS.batch_size))
    handle_train, handle_val = sess.run([model.iter_train_handle, model.iter_val_handle])
    # generate batches and run graph
    for epoch in range(0, FLAGS.max_epochs):
        t0 = time.time()
        for i in range(0, epoch_size):
            sess.run([model.loss, model.train_op], feed_dict={model.handle: handle_train}, options=options,
                     run_metadata=run_metadata)
            if i % 100 == 0:
                err, summaries = sess.run(
                    [model.loss, model.summaries], feed_dict={model.handle: handle_train}, options=options,
                    run_metadata=run_metadata)
                t1 = time.time()
                [train_writer.add_summary(s, epoch * epoch_size + i) for s in summaries]
                saver.save(sess, os.path.join("name", 'model'))
                if FLAGS.debug:
                    train_writer.add_run_metadata(run_metadata, "step{}".format(epoch * epoch_size + i),
                                                  global_step=None)
                train_writer.flush()
                print("Loss:{}".format(err))
                print("{} sec per sample".format((t1 - t0) / (100 * FLAGS.batch_size)))
                if FLAGS.debug:
                    return
        print("Validating...")
        for i in range(0, validation_size):
            sess.run(model.val_update, feed_dict={model.handle: handle_val}, options=options,
                     run_metadata=run_metadata)
        validation_summary = sess.run(model.validation_summary, options=options, run_metadata=run_metadata)
        sess.run(model.reset_mean)
        train_writer.add_summary(validation_summary, epoch)
        train_writer.flush()
    print("finished")
    train_writer.close()
    sess.close()
