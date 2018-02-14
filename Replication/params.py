import tensorflow as tf
tf.app.flags.DEFINE_float('learning-rate', 0.001, 'Learning Rate. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 50, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 4, 'BatchSize. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('debug', False, 'Batch Size. (default: %(default)d)')
FLAGS = tf.app.flags.FLAGS