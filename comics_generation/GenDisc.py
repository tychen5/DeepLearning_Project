import tensorflow as tf
from tensorflow import layers


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


# WGAN/DCGAN
class Generator(object):

    def __init__(self, z_dim, h_dim, generator_output_layer):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = [None, 64, 64, 3]
        self.name = 'dcgan/g_net'
        self.generator_output_layer = generator_output_layer

    def __call__(self, z, h, training):
        with tf.variable_scope(self.name) as vs:
            z_h = tf.concat([z, h], axis=1)

            fc1 = layers.dense(
                z_h, 64 * 8 * 4 * 4,
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            fc1_batch_norm = layers.batch_normalization(
                fc1, training=training
            )

            fc1_a = tf.nn.relu(fc1_batch_norm)

            conv = tf.reshape(fc1_a, [-1, 4, 4, 64 * 8])

            conv1 = layers.conv2d_transpose(
                conv, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv1_batch_norm = layers.batch_normalization(
                conv1, training=training
            )
            conv1_a = tf.nn.relu(conv1_batch_norm)

            conv2 = layers.conv2d_transpose(
                conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv2_batch_norm = layers.batch_normalization(
                conv2, training=training
            )
            conv2_a = tf.nn.relu(conv2_batch_norm)

            conv3 = layers.conv2d_transpose(
                conv2_a, 64, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            conv3_batch_norm = layers.batch_normalization(
                conv3, training=training
            )
            conv3_a = tf.nn.relu(conv3_batch_norm)

            conv4 = layers.conv2d_transpose(
                conv3_a, 3, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                activation=None
            )

            return tf.nn.tanh(conv4)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# dcgan
class Discriminator(object):

    def __init__(self, h_dim):
        self.x_dim = [None, 64, 64, 3]
        self.h_dim = h_dim
        self.name = 'dcgan/d_net'

    def __call__(self, x, h, training, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            conv1 = layers.conv2d(
                x, 64, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )
            # normalization
            conv1_batch_norm = layers.batch_normalization(
                conv1, training=training
            )
            conv1_a = leaky_relu(conv1_batch_norm)

            conv2 = layers.conv2d(
                conv1_a, 64 * 2, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # normalization
            conv2_batch_norm = layers.batch_normalization(
                conv2, training=training
            )
            conv2_a = leaky_relu(conv2_batch_norm)

            conv3 = layers.conv2d(
                conv2_a, 64 * 4, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # normalization
            conv3_batch_norm = layers.batch_normalization(
                conv3, training=training
            )
            conv3_a = leaky_relu(conv3_batch_norm)

            conv4 = layers.conv2d(
                conv3_a, 64 * 8, kernel_size=[4, 4], strides=[2, 2],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # normalization
            conv4_batch_norm = layers.batch_normalization(
                conv4, training=training
            )
            conv4_a = leaky_relu(conv4_batch_norm)

            r_h = tf.tile(tf.reshape(h, [-1, 1, 1, self.h_dim]), [1, 4, 4, 1])

            concat_h = tf.concat([conv4_a, r_h], axis=3)

            conv5 = layers.conv2d(
                concat_h, 64 * 8, kernel_size=[1, 1], strides=[1, 1],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            # normalization
            conv5_batch_norm = layers.batch_normalization(
                conv5, training=training
            )
            conv5_a = leaky_relu(conv5_batch_norm)

            conv6 = layers.conv2d(
                conv5_a, 1, kernel_size=[4, 4], strides=[1, 1],
                padding='valid',
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                activation=None
            )

            conv6_sq = tf.squeeze(conv6, [1, 2, 3])

            return conv6_sq

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
