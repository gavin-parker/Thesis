import tensorflow as tf
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')
FLAGS = tf.app.flags.FLAGS

def conv2d_extraction(x, filters, size):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_size=size)

def conv2d_reconstruction(x, filters, size):
    return tf.layers.conv2d(inputs=x,
                            filters=filters,
                            kernel_initializer=tf.random_uniform_initializer(-0.05, 0.05),
                            padding='SAME',
                            activation=tf.identity,
                            kernel_size=size)

class Generator:
    def __init__(self, sess):
        self.sess = sess
        envmap_size, input_size = (1536,256), (1080,1920)
        self.envmaps = tf.placeholder(tf.float32, [None, envmap_size[0], envmap_size[1], 3])
        self.renders = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])

        self.pred = self.generate(self.renders)
        self.loss = tf.reduce_mean(tf.square(self.envmaps - self.pred))
        self.saver = tf.train.Saver()
        return

    #Convolutional graph definition for render -> envmap
    def generate(self, x_image):
        conv1 = conv2d_extraction(x_image, 64, [1,9,9])
        conv2 = conv2d_extraction(conv1, 32, [64,1,1])
        conv3 = conv2d_reconstruction(conv2, 1, [32,5,5])
        return conv3

    def train(self, dataset):
        self.train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()
        #generate batches and run graph

