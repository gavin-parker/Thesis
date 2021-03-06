from trainer.params import FLAGS
import tensorflow as tf

from tensorflow.python import debug as tf_debug
import time
import glob
import os
import cv2
import numpy as np
import math
from skimage.measure import compare_ssim as ssim
from tensorflow.python.lib.io import file_io
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
    epoch_size = len(file_io.get_matching_files("{}/left/*.png".format(FLAGS.train_dir)))
    epoch_size /= FLAGS.batch_size
    validation_size = len(file_io.get_matching_files("{}/left/*.png".format(FLAGS.val_dir)))
    validation_size /= FLAGS.batch_size
    print("beginning training with learning rate: {}".format(FLAGS.learning_rate))
    print("Epoch size: {}".format(epoch_size))
    print("Batch size: {}".format(FLAGS.batch_size))
    handle_train, handle_val = sess.run([model.iter_train_handle, model.iter_val_handle])
    val_best = 9999999999
    stop_count=0
    # generate batches and run graph
    for epoch in range(0, FLAGS.max_epochs):
        if stop_count > 5:
            break
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
        validation_summary, val_ssim = sess.run([model.validation_summary, model.val_loss], options=options, run_metadata=run_metadata)
        if val_ssim < val_best:
            saver.save(sess, "{}/{}/model".format(FLAGS.train_dir, name))
            val_best = val_ssim
        else:
            stop_count += 1
        sess.run(model.reset_mean)
        train_writer.add_summary(validation_summary, epoch)
        train_writer.flush()
    print("finished")
    train_writer.close()
    sess.close()


def collect_results(model=None):
    assert model

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.Session(config=config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    tf.train.start_queue_runners(sess=sess)

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.test_model_dir)
    left_samples = file_io.get_matching_files("{}/left/*.png".format(FLAGS.val_dir))
    right_samples = file_io.get_matching_files("{}/right/*.png".format(FLAGS.val_dir))
    gt_samples = file_io.get_matching_files("{}/envmaps/*.hdr".format(FLAGS.val_dir))
    bg_samples = file_io.get_matching_files("{}/bg/*.png".format(FLAGS.val_dir))
    norm_samples = file_io.get_matching_files("{}/norms/*.png".format(FLAGS.val_dir))

    total_time = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    if not os.path.exists("{}/results".format(FLAGS.val_dir)):
        os.makedirs("{}/results".format(FLAGS.val_dir))
    with open('{}/results/meta.csv'.format(FLAGS.val_dir), 'w+') as f:
        for (l, r, gt, bg, norms) in zip(left_samples, right_samples, gt_samples, bg_samples, norm_samples):
            left, right, envmap, background, norms = prep_images(l, r, gt, bg, norms)
            t0 = time.time()
            prediction,test = sess.run([model.converted_prediction, model.test],
                                  feed_dict={model.left_image: left,
                                             model.right_image: right,
                                             model.bg_image: background})
            prediction = prediction[0]
            print(np.mean(prediction))
            t1 = time.time()
            mse = mean_squared_error(envmap, prediction)

            ss = (1 - ssim(envmap, prediction, multichannel=True)) / 2
            #print(ss)

            total_ssim += ss
            total_mse += mse
            name = os.path.splitext(os.path.basename(l))[0]
            gt_mag = np.linalg.norm(envmap, axis=-1)
            pred_mag = np.linalg.norm(prediction, axis=-1)
            gt_sun = np.unravel_index(gt_mag.argmax(), gt_mag.shape)
            pred_sun = np.unravel_index(pred_mag.argmax(), pred_mag.shape)
            x_diff = min(math.fabs(gt_sun[0] - pred_sun[0]), math.fabs(gt_sun[0]+64-pred_sun[0]))
            y_diff = min(math.fabs(gt_sun[1] - pred_sun[1]), math.fabs(gt_sun[1]+64-pred_sun[1]))
            sun_dist = math.sqrt(x_diff*x_diff + y_diff*y_diff)
            f.write("{},{},{},{}\n".format(name,mse, ss, sun_dist))
            #print("inference time: {}".format(t1-t0))
            total_time += (t1 - t0)
            cv2.imwrite('{}/results/{}.hdr'.format(FLAGS.val_dir, name), prediction)
    print("Avg inference time: {}".format(total_time / len(left_samples)))
    print("Total MSE: {}".format(total_mse / len(left_samples)))
    print("Total DSSIM: {}".format(total_ssim / len(left_samples)))

    sess.close()


def mean_squared_error(a, b):
    error = a - b
    squared_error = error * error
    return np.mean(squared_error, axis=(0, 1, 2))

def prep_col_img(im):
    im = cv2.imread(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256)).astype(np.float32)
    im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    im = np.expand_dims(im, axis=0)
    return im

def prep_images(l, r, gt, bg, norms):
    left = prep_col_img(l)
    right = prep_col_img(r)
    bg = prep_col_img(bg)
    norms = prep_col_img(norms)
    envmap = cv2.imread(gt, cv2.IMREAD_UNCHANGED)
    return left, right, envmap, bg, norms

