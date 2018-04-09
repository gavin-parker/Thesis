import tensorflow as tf

tf.app.flags.DEFINE_float('learning-rate', 1.6e-7, 'Learning Rate. (default: %(default)d)')
tf.app.flags.DEFINE_integer('max-epochs', 50, 'Number of epochs to run. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 4, 'BatchSize. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('debug', False, 'Enable debug options. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('profile', False, 'Enable profile options. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('validate', False, 'Operate on validation set. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('test', False, 'Run test renders. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('real', False, 'Operate on real data set. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('fine-tune', False, 'Pre-load previous best. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('lab-space', True, 'Use CIE Lab space. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('siamese', True, 'Used shared weights in conv layers. (default: %(default)d)')
tf.app.flags.DEFINE_boolean('app', False, 'Dont load train data. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', 'envmap_synthetic_results', 'Output Log directory')
tf.app.flags.DEFINE_string('test-model-dir', 'best_dematerial', 'Directory of model for testing')
tf.app.flags.DEFINE_float('weight-decay', 0.0005, 'Weight decay factor.')
tf.app.flags.DEFINE_string('train-dir', "/mnt/black/MultiNatIllum/data/single_material_multiple_objects/singlets/",
                           'Path to training set')
tf.app.flags.DEFINE_string('val-dir', "/mnt/black/MultiNatIllum/data/single_material_multiple_objects/singlets/",
                           'Path to validation set')
FLAGS = tf.app.flags.FLAGS
