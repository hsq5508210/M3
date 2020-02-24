#-*-coding:utf-8-*-
import tensorflow as tf
from utils import compute_loss
from tensorflow.python.platform import flags
import tensorflow as tf
import cv2
import numpy as np
from model import Conv64F
from time import time
np.random.seed(0)
query_set = np.random.normal(size=(10, 3))
s_modal = tf.convert_to_tensor([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]], dtype=tf.float32)
q_modal = tf.convert_to_tensor([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]], dtype=tf.float32)
s_label = tf.convert_to_tensor([[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1]], dtype=tf.float32)
q_label = tf.convert_to_tensor([[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1]], dtype=tf.float32)
q = np.random.normal(size=(1, 3))
q = tf.convert_to_tensor(q, dtype=tf.float32)
support_set = np.random.normal(size=(20, 3))
query_num = 5
support__num = 10
vector_dim = 2
q_test = tf.convert_to_tensor(np.random.normal(size=(query_num, vector_dim)), dtype=tf.float32)
s_test = tf.convert_to_tensor(np.random.normal(size=(support__num, vector_dim)), dtype=tf.float32)
distance_style = 'euc_v1'
usehard = False

# input = np.reshape(np.resize(cv2.imread('dog.jpg')/255.0, (84, 84, 3)), (1, 84, 84 ,3))
# print(input.shape)
# conv64f = Conv64F()
#
#
# output = conv64f(input)
# print(output.shape)
# print(output)

print(q_test.shape)
print(q_test)
q_test1 = tf.reshape(tf.tile(q_test, [1, s_test.shape[0]]), (query_num, support__num, vector_dim))

print(q_test1)
print(q_test1.shape)
print(s_test.shape)
print(s_test)
# s_test = tf.reshape(tf.tile(s_test, [1, q_test.shape[0]]), (query_num, support__num, vector_dim))
s_test1 = tf.tile(input=tf.reshape(s_test, (1, -1, vector_dim)), multiples=[q_test.shape[0], 1, 1])
def euc_v1(x, y):
    width = y.shape[0]
    high = x.shape[0]
    x_2 = tf.reshape(tf.reduce_sum(x * x, axis=1), (-1, 1))
    y_2 = tf.reshape(tf.reduce_sum(y * y, axis=1), (1, -1))
    x_fill_op = tf.ones((1, width), dtype=tf.float32)
    y_fill_op = tf.ones((high, 1), dtype=tf.float32)
    xy = 2.0 * tf.matmul(x, tf.transpose(y))
    # res = tf.matmul(x_2, x_fill_op) + tf.matmul(y_fill_op, y_2) - xy
    distance = tf.sqrt(
        tf.nn.relu(tf.matmul(x_2, x_fill_op) + tf.matmul(y_fill_op, y_2) - xy))
    return distance
def euc_v2(x, y):
    width, dim = y.shape
    high, dim = x.shape
    x_tile = tf.reshape(tf.tile(x, [1, width]), (high, width, dim))
    y_tile = tf.tile(input=tf.reshape(y, (1, -1, dim)), multiples=[high, 1, 1])
    distance = tf.sqrt(tf.reduce_sum(tf.square(x_tile - y_tile), axis=2))
    return distance
# print(s_test1)

a = [1, 2, 3]
ad = tf.linalg.diag(a)
print(ad)



