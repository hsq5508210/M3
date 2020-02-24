#-*-coding:utf-8-*-
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import Model
import tensorflow as tf
import math
from tensorflow.python.platform import flags
import utils
from utils import conv_block, get_acc, loss_eps, category_choose, category_loss, get_prototype, vectorlize, log_liklyhood, scd, support_weight, intra_var, inter_var, loss_eps_prototype
FLAGS = flags.FLAGS



class M3:
    def __init__(self):
        super(M3, self).__init__()
        if FLAGS.backbone == 'Conv64F':
            if not FLAGS.load_model:
                self.backbone = Conv64F()
        if FLAGS.loss_function == 'mse':
            self.loss_function = category_loss
        elif FLAGS.loss_function == 'log':
            self.loss_function = log_liklyhood
        if FLAGS.prototype:
            self.eps_loss = loss_eps_prototype
        else:
            self.eps_loss = loss_eps
        self.optimizer = tf.optimizers.Adam(FLAGS.lr)
        self.lr = FLAGS.lr
    def __call__(self, query_set, support_set, train='train'):
        with tf.name_scope('loss_function'):
            qx, qy, qm = query_set
            sx, sy, sm = support_set
            output_s = vectorlize(self.backbone(sx))
            output_q = vectorlize(self.backbone(qx))
            if FLAGS.prototype:
                prototype_s, prototype_sy = get_prototype(output_s, sy)
            predict = category_choose(output_q, prototype_s, prototype_sy)
            accs = get_acc(predict, qy)
            losses = classify_loss = self.loss_function(predict, qy)
            if FLAGS.eps_loss and train == 'train':
                epsloss = self.eps_loss(output_s, output_q, sm, qm, sy, qy, FLAGS.margin)
                losses = classify_loss * (1.0 - FLAGS.weight) + FLAGS.weight * epsloss
            loss = tf.reduce_mean(losses)
            if train == 'train':
                self.trainop(loss)
            return [loss, accs]
    def trainop(self, loss):
        self.optimizer.minimize(loss)




class Conv64F(Model):
    def __init__(self):
        super(Conv64F, self).__init__()
        self.conv1 = conv_mp_layer('1')
        self.conv2 = conv_mp_layer('2')
        self.conv3 = conv_layer('3')
        self.conv4 = conv_layer('4')

    def call(self, x):
        with tf.name_scope('conv64f_forward'):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        return x

class conv_mp_layer(tf.keras.layers.Layer):
    def __init__(self, name_scope):
        super(conv_mp_layer, self).__init__()
        # self.name_scope = name_scope
    def call(self, input):
        with tf.name_scope('conv_maxpool'):
            conv = tf_layers.Conv2D(64, 3, (1, 1), 'same', use_bias=True)(input)
            norm = tf_layers.BatchNormalization()(conv)
            act = tf_layers.LeakyReLU(0.2)(norm)
            mp = tf_layers.MaxPool2D(strides=2)(act)
        return mp
class conv_layer(tf.keras.layers.Layer):
    def __init__(self, name_scope):
        super(conv_layer, self).__init__()
        # self.name_scope = name_scope
    def call(self, input):
        with tf.name_scope('conv'):
            conv = tf_layers.Conv2D(64, 3, (1, 1), 'same', use_bias=True)(input)
            norm = tf_layers.BatchNormalization()(conv)
            act = tf_layers.LeakyReLU(0.2)(norm)
        return act




