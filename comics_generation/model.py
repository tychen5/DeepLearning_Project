import tensorflow as tf
from GenDisc import Discriminator, Generator


# imporved WGAN
class ConditionalWassersteinGAN(object):

    def __init__(self, z_dim, h_dim, learning_rate, scale, generator_output_layer):
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.g_net = Generator(z_dim, h_dim, generator_output_layer)
        self.d_net = Discriminator(h_dim)

        self.training = tf.placeholder(tf.bool, [])

        self.with_text = tf.placeholder(tf.float32, [None])

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        self.h = tf.placeholder(tf.float32, [None, h_dim])
        self.h_ = tf.placeholder(tf.float32, [None, h_dim])

        # 錯的圖片
        self.x_ = self.g_net(self.z, self.h, self.training)

        # 對的圖片隊的敘述
        self.d = self.d_net(self.x, self.h, self.training, reuse=False)

        # 錯的圖對的敘述
        self.d_ = self.d_net(self.x_, self.h, self.training)
        self.d_w_ = self.d_net(self.x_w_, self.h, self.training)

        # 對的圖錯的敘述
        self.d_h_ = self.d_net(self.x, self.h_, self.training)

        self.g_loss = - tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - (1 * tf.reduce_mean(self.d_) + 1 * tf.reduce_mean(self.d_h_)
                                                + 1 * tf.reduce_mean(self.d_w_)) / (1 + 1 + 1)

        # penalty for "improved wgan"
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat, self.h, self.training)

        dx = tf.gradients(d_hat, x_hat)[0]
        dx_norm = tf.sqrt(tf.reduce_sum(tf.square(dx), axis=[1, 2, 3]))

        ddx = scale * tf.reduce_mean(tf.square(dx_norm - 1.0))

        self.d_loss = -(self.d_loss - ddx)

        self.d_opt, self.g_opt = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
            # self.d_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.d_loss,
            # var_list=self.d_net.vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars)
            # self.g_opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.g_loss,
            # var_list=self.g_net.vars)


# DCGAN
class ConditionalGAN(object):

    def __init__(self, z_dim, h_dim, learning_rate, scale, generator_output_layer):
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.g_net = Generator(z_dim, h_dim, generator_output_layer)
        self.d_net = Discriminator(h_dim)

        self.training = tf.placeholder(tf.bool, [])

        self.with_text = tf.placeholder(tf.float32, [None])

        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.x_w_ = tf.placeholder(tf.float32, [None, 64, 64, 3])

        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        # 對的描述
        self.h = tf.placeholder(tf.float32, [None, h_dim])
        # 錯的描述
        self.h_ = tf.placeholder(tf.float32, [None, h_dim])

        # 錯的圖片
        self.x_ = self.g_net(self.z, self.h, self.training)

        self.d = self.d_net(self.x, self.h, self.training, reuse=False)

        # 錯圖對描述
        self.d_ = self.d_net(self.x_, self.h, self.training)

        # 爛圖對描述
        self.d_w_ = self.d_net(self.x_w_, self.h, self.training)

        # 好圖錯敘述
        self.d_h_ = self.d_net(self.x, self.h_, self.training)

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.ones_like(self.d_)))

        self.d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) + (tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_, labels=tf.zeros_like(self.d_))) +
                                                                                                    tf.reduce_mean(
                                                                                                        tf.nn.sigmoid_cross_entropy_with_logits(
                                                                                                            logits=self.d_w_,
                                                                                                            labels=tf.zeros_like(
                                                                                                                self.d_w_))) +
                                                                                                    tf.reduce_mean(
                                                                                                        tf.nn.sigmoid_cross_entropy_with_logits(
                                                                                                            logits=self.d_h_,
                                                                                                            labels=tf.zeros_like(
                                                                                                                self.d_h_)))) / 3

        self.d_opt, self.g_opt = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9) \
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9) \
                .minimize(self.g_loss, var_list=self.g_net.vars)
