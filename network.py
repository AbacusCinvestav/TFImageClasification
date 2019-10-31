import numpy as np
import tensorflow as tf

def convolutional_layer(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool_layer(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def cnn_arquitecture(x, weights, biases):

    conv1 = convolutional_layer(x, weights['wc1'], biases['bc1']);
    pool1 = maxpool_layer(conv1, k=2)

    conv2 = convolutional_layer(pool1, weights['wc2'], biases['bc2'])
    pool2 = maxpool_layer(conv2, k=2)

    conv3 = convolutional_layer(pool2, weights['wc3'], biases['bc3'])
    pool3 = maxpool_layer(conv3, k=2)

    conv4 = convolutional_layer(pool3, weights['wc4'], biases['bc4'])
    pool4 = maxpool_layer(conv4, k=2)

    flat1 = tf.reshape(pool4, [-1, weights['wd1'].get_shape().as_list()[0]])
    flat1 = tf.add(tf.matmul(flat1, weights['wd1']), biases['bd1'])
    flat1 = tf.nn.relu(flat1)

    flat2 = tf.add(tf.matmul(flat1, weights['wd2']), biases['bd2'])
    flat2 = tf.nn.relu(flat2)

    out = tf.add(tf.matmul(flat2, weights['out'], name='y_pred'), biases['out'])

    return out

def cnn_model(n_input, n_classes, channels, x, y):

    weights = {
        'wc1': tf.get_variable('W0', shape=(3,3,channels,32), initializer=tf.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
        'wc4': tf.get_variable('W3', shape=(3,3,128,128), initializer=tf.contrib.layers.xavier_initializer()),
        'wd1': tf.get_variable('W4', shape=(8*8*128,128), initializer=tf.contrib.layers.xavier_initializer()),
        'wd2': tf.get_variable('W5', shape=(128,128), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'bc4': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'bd2': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B6', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer())
    }

    pred = cnn_arquitecture(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

    y_true_cls = tf.argmax(y, axis=1)
    y_pred_cls = tf.argmax(pred, axis=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    return {'pred':pred, 'cost':cost, 'optimizer':optimizer, 'accuracy':accuracy}
