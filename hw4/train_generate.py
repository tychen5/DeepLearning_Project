import tensorflow as tf
import os
import numpy as np
import sys

from model import ConditionalGAN
from preprocess import DataManager
from scipy.misc import imsave

#arg1 = sys.argv[1]

tf.flags.DEFINE_integer("mode", 1, "0=train , 1=test")
tf.flags.DEFINE_string("model_type", "dcgan", " dcgan /  wgan")
tf.flags.DEFINE_string("test_text", "./testing_text.txt", "testing text path")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
FLAGS.generator_output_layer = "tanh"
FLAGS.scale = 10.0  # wgan penalty
FLAGS.learning_rate = 0.0002
FLAGS.tag_file = "./tags_clean.csv"
FLAGS.load_dir = "./models"
FLAGS.model_type = "dcgan"  # wgan
FLAGS.load = True  # False
FLAGS.valid_img_dir = "./validation/"
FLAGS.test_img_dir = "./samples/"
FLAGS.log = "./models"
FLAGS.seed = 123  # random
FLAGS.vocab = "./vocab"
FLAGS.z_type = "normal"
#FLAGS.test_text = "./testing_text.txt"  # 要改成使用者參數輸入 # str(arg1)
FLAGS.img_dir = "data/faces"
FLAGS.epochs = 600
FLAGS.d_epochs = 1  # 5
FLAGS.g_epochs = 1
FLAGS.batch_size = 256
FLAGS.z_dim = 100

if FLAGS.seed:
    np.random.seed(seed=FLAGS.seed)


def run_train_epoch(epoch, sess, dm, model):
    total_batch_num = dm.total_batch_num(FLAGS.batch_size)
    total_d_loss = 0
    d_count = 0
    total_g_loss = 0
    g_count = 0
    for i in range(total_batch_num):

        data, bz, bh_, bx_w_ = dm.draw_batch(FLAGS.batch_size, FLAGS.z_dim, mode='train')
        bx = [d.img for d in data]
        bh = [d.tags for d in data]
        bwith_text = [d.with_text for d in data]

        if i % 1 == 0:
            for d_i in range(FLAGS.d_epochs):
                _, d_loss = sess.run([model.d_opt, model.d_loss],
                                     feed_dict={
                                         model.x: bx,
                                         model.z: bz,
                                         model.h: bh,
                                         model.h_: bh_,
                                         model.x_w_: bx_w_,
                                         model.training: True,
                                         model.with_text: bwith_text
                                     }
                                     )
                total_d_loss += d_loss
                d_count += 1

        for g_i in range(FLAGS.g_epochs):
            _, g_loss = sess.run([model.g_opt, model.g_loss],
                                 feed_dict={
                                     model.x: bx,
                                     model.z: bz,
                                     model.h: bh,
                                     model.h_: bh_,
                                     model.x_w_: bx_w_,
                                     model.training: True,
                                     model.with_text: bwith_text
                                 }
                                 )
            total_g_loss += g_loss
            g_count += 1

    return total_d_loss / d_count, total_g_loss / g_count


def run_valid(epoch, sess, dm, model):
    if not os.path.exists(FLAGS.valid_img_dir):
        os.makedirs(FLAGS.valid_img_dir)

    data, bz = dm.draw_batch(10, FLAGS.z_dim, mode='random')  # 隨機找圖片出來
    bh = [d.tags for d in data]
    images = sess.run(model.x_,
                      feed_dict={
                          model.z: bz,
                          model.h: bh,
                          model.training: False
                      }
                      )

    # save images
    for image, d in zip(images, data):
        image = (image + 1.0) / 2.0

        img_resized = image

        tag_text = d.tag_text
        img_id = d.img_id
        img_filename = "{}_{}_{}.jpg".format(epoch, img_id, tag_text)
        imsave(os.path.join(FLAGS.valid_img_dir, img_filename), img_resized)

    data, bz = dm.draw_batch(10, FLAGS.z_dim, mode='gif')
    bh = [d.tags for d in data]
    images = sess.run(model.x_,
                      feed_dict={
                          model.z: bz,
                          model.h: bh,
                          model.training: False
                      }
                      )

    for image, d in zip(images, data):  # 儲存圖片

        image = (image + 1.0) / 2.0

        img_resized = image

        tag_text = d.tag_text
        img_id = d.img_id
        img_filename = "GIF_{}_{}_{}.jpg".format(img_id, tag_text, epoch)
        imsave(os.path.join(FLAGS.valid_img_dir, img_filename), img_resized)


def train():
    dm = DataManager(FLAGS.mode,
                     FLAGS.tag_file, FLAGS.img_dir, FLAGS.test_text, FLAGS.vocab, FLAGS.z_dim, FLAGS.z_type,
                     FLAGS.generator_output_layer)
    from model import ConditionalWassersteinGAN
    from model import ConditionalGAN
    with tf.Graph().as_default():

        if FLAGS.model_type == 'wgan':

            model = ConditionalWassersteinGAN(
                FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

        elif FLAGS.model_type == 'dcgan':

            model = ConditionalGAN(
                FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        saver = tf.train.Saver(max_to_keep=300)

        def load_pretrain(sess):
            saver.restore(sess, os.path.join(FLAGS.log, "checkpoint"))

        if FLAGS.load:
            sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver, init_fn=load_pretrain)
            print("==loaded model==")
        else:
            sv = tf.train.Supervisor(logdir=FLAGS.log, saver=saver)

        with sv.managed_session(config=config) as sess:

            for epoch in range(FLAGS.epochs):  # 產生圖片給人工審閱
                d_loss, g_loss = run_train_epoch(epoch, sess, dm, model)
                info_output = "EPOCH: {} disc_loss: {}, gen_loss: {}".format(epoch, d_loss, g_loss)
                print(info_output)
                run_valid(epoch, sess, dm, model)
                saver.save(sess, save_path=FLAGS.log, global_step=epoch)


def run_test_epoch(sess, dm, model):
    if not os.path.exists(FLAGS.test_img_dir):
        os.makedirs(FLAGS.test_img_dir)

    print("Images saved in \"samples\" dir")

    total_batch_num = dm.total_batch_num(FLAGS.batch_size, mode='test')

    for i in range(total_batch_num):

        data, bz = dm.draw_batch(FLAGS.batch_size, FLAGS.z_dim, mode='test')
        bh = [d.tags for d in data]

        images = sess.run(model.x_,
                          feed_dict={
                              model.z: bz,
                              model.h: bh,
                              model.training: False
                          }
                          )

        for _, (image, d) in enumerate(zip(images, data)):
            image = (image + 1.0) / 2.0

            img_resized = image

            img_id = d.img_id
            img_filename = "sample_{}.jpg".format(img_id)
            print(img_filename)
            imsave(os.path.join(FLAGS.test_img_dir, img_filename), img_resized)


def test():
    FLAGS.vocab = "./vocab"
    dm = DataManager(FLAGS.mode,
                     FLAGS.tag_file, FLAGS.img_dir, FLAGS.test_text, FLAGS.vocab, FLAGS.z_dim, FLAGS.z_type,
                     FLAGS.generator_output_layer)

    with tf.Graph().as_default():
        model = ConditionalGAN(
            FLAGS.z_dim, 2 * dm.vocab.vocab_size, FLAGS.learning_rate, FLAGS.scale, FLAGS.generator_output_layer)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        saver = tf.train.Saver(max_to_keep=300)

        with tf.Session() as sess:
            saver.restore(sess, os.path.join(FLAGS.log, "-61"))  # model name
            run_test_epoch(sess, dm, model)


if __name__ == "__main__":

    if FLAGS.mode % 2 == 0:
        train()

    if FLAGS.mode // 2 == 0:
        test()
