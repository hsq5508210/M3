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
            if not FLAGS.load_ckpt:
                self.backbone = Conv64F()
        if FLAGS.loss_function == 'mse':
            self.loss_function = category_loss
        elif FLAGS.loss_function == 'log':
            self.loss_function = log_liklyhood
        if FLAGS.prototype:
            self.eps_loss = loss_eps_prototype
        else:
            self.eps_loss = loss_eps
        self.lr = FLAGS.lr
    def __call__(self, query_set, support_set, train='train'):
        with tf.name_scope('loss_function'):
            qx, qy, qm = tf.cast(query_set[0], dtype=tf.float32), tf.cast(query_set[1], dtype=tf.float32), tf.cast(query_set[2], dtype=tf.float32)
            sx, sy, sm = tf.cast(support_set[0], dtype=tf.float32), tf.cast(support_set[1], dtype=tf.float32), tf.cast(support_set[2], dtype=tf.float32)

            if train == 'train':
                with tf.GradientTape() as grad_tape:
                    output_s = vectorlize(self.backbone(sx))
                    output_q = vectorlize(self.backbone(qx))
                    if FLAGS.prototype:
                        prototype_s, prototype_sy = get_prototype(output_s, sy)
                    predict = category_choose(output_q, prototype_s, prototype_sy)
                    classify_loss = self.loss_function(predict, qy)
                    if FLAGS.eps_loss:
                        epsloss = self.eps_loss(output_s, output_q, sm, qm, sy, qy, FLAGS.margin)
                        losses = classify_loss * (1.0 - FLAGS.loss_weight) + FLAGS.loss_weight * epsloss
                    else:
                        losses = classify_loss
                    loss = tf.reduce_mean(losses)
                    accs = get_acc(predict, qy)
                grads = grad_tape.gradient(loss, self.backbone.trainable_variables)
                # print(grads)
                self.optmizer = tf.optimizers.Adam(self.lr)
                self.optmizer.apply_gradients(zip(grads, self.backbone.trainable_variables))
            else:
                loss, accs = self.test(qx, qy, sx, sy)
        return [loss, accs]
    def decay(self):
        iter = FLAGS.decay_iteration
        decay_rate = FLAGS.decay_rate
        self.lr = self.lr * (decay_rate ** float(i // iter))
    def test(self, qx, qy, sx, sy):
        def test_acc(inp):
            with tf.name_scope("test_acc"):
                support_x, support_y, query_x, query_y = inp
                output_s = vectorlize(self.backbone(support_x))
                output_q = vectorlize(self.backbone(query_x))
                if FLAGS.prototype:
                    prototype_s, prototype_sy = get_prototype(output_s, support_y)
                predict = category_choose(output_q, prototype_s, prototype_sy)
                accurcy = get_acc(predict, query_y)
            return accurcy
        def test_loss(inp):
            with tf.name_scope("test_acc"):
                support_x, support_y, query_x, query_y = inp
                output_s = vectorlize(self.backbone(support_x))
                output_q = vectorlize(self.backbone(query_x))
                if FLAGS.prototype:
                    prototype_s, prototype_sy = get_prototype(output_s, support_y)
                predict = category_choose(output_q, prototype_s, prototype_sy)
                classify_loss = self.loss_function(predict, query_y)
            return tf.reduce_mean(classify_loss)
        batchinp = sx, sy, qx, qy
        acc = tf.map_fn(fn=test_acc, elems=(batchinp), dtype=tf.float32, parallel_iterations=FLAGS.test_batch_size)
        loss = tf.map_fn(fn=test_loss, elems=(batchinp), dtype=tf.float32, parallel_iterations=FLAGS.test_batch_size)
        return loss, acc







class Conv64F(Model):
    def __init__(self):
        super(Conv64F, self).__init__()
        self.conv1 = conv_mp_layer('1')
        self.conv2 = conv_mp_layer('2')
        self.conv3 = conv_layer('3')
        self.conv4 = conv_layer('4')
        self.weights_list = [self.conv1.weights, self.conv2.weights, self.conv3.weights, self.conv4.weights]

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
        self.initializer = tf.initializers.GlorotUniform(seed=5)
        self.conv = tf_layers.Conv2D(64, 3, (1, 1), 'same', use_bias=True, kernel_initializer=self.initializer,
                                trainable=True)
        self.norm = tf_layers.BatchNormalization()
        self.act = tf_layers.LeakyReLU(0.2)
        self.mp = tf_layers.MaxPool2D(strides=2)
    def call(self, input):
        with tf.name_scope('conv_maxpool'):
            conv_out = self.conv(input)
            norm_out = self.norm(conv_out)
            act_out = self.act(norm_out)
            output = self.mp(act_out)
        return output
class conv_layer(tf.keras.layers.Layer):
    def __init__(self, name_scope):
        super(conv_layer, self).__init__()
        self.initializer = tf.initializers.GlorotUniform(seed=5)
        self.conv = tf_layers.Conv2D(64, 3, (1, 1), 'same', use_bias=True, kernel_initializer=self.initializer,
                                trainable=True)
        self.norm = tf_layers.BatchNormalization()
        self.act = tf_layers.LeakyReLU(0.2)
        # self.name_scope = name_scope
    def call(self, input):
        with tf.name_scope('conv'):
            conv_out = self.conv(input)
            norm_out = self.norm(conv_out)
            output = self.act(norm_out)
        return output




