import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class FeatEmbed:
    def __init__(self, lstm_size, img_dim, name):
        with tf.variable_scope(name):
            self.lstm_size = lstm_size
            self.img_dim = img_dim

            # FC transform feat to image
            self.embed_W1 = weight_variable(shape=[self.img_dim, self.lstm_size])
            self.embed_B1 = weight_variable(shape=[self.lstm_size])
            self.embed_W2 = weight_variable(shape=[self.lstm_size, self.lstm_size])
            self.embed_B2 = weight_variable(shape=[self.lstm_size])

    def __call__(self, feat):
        h1 = tf.nn.tanh(tf.matmul(feat, self.embed_W1) + self.embed_B1)
        output = tf.nn.tanh(tf.matmul(h1, self.embed_W2) + self.embed_B2)
        return output


class Att:
    def __init__(self, lstm_size, img_dim, name):
        with tf.variable_scope(name):
            self.lstm_size = lstm_size
            self.img_dim = img_dim

            # Channel-wise attention coefficients
            self.ch_W1 = weight_variable(shape=[self.img_dim, self.lstm_size])
            self.ch_prevW = weight_variable(shape=[self.lstm_size, self.lstm_size])
            self.ch_B1 = bias_variable(shape=[self.lstm_size])

            # FC to softmax
            self.ch_W2 = weight_variable(shape=[self.lstm_size, self.img_dim])
            self.ch_B2 = bias_variable(shape=[self.img_dim])

    def __call__(self, features, prev_word, sent_length):
        trans_feat = tf.tile(tf.expand_dims(features, 1), multiples=[1, sent_length, 1])
        trans_feat = tf.reshape(trans_feat, shape=[-1, self.img_dim])

        # channel attention
        ch_att_image1 = tf.matmul(trans_feat, self.ch_W1)
        print('ch_att_image dim:', ch_att_image1.get_shape())
        ch_att_prev1 = tf.matmul(prev_word, self.ch_prevW)
        print('ch_att_prev dim:', ch_att_prev1.get_shape())
        ch_att_concat = tf.tanh(tf.add(ch_att_image1, ch_att_prev1) + self.ch_B1)

        # FC to softmax
        fc_att = tf.tanh(tf.matmul(ch_att_concat, self.ch_W2) + self.ch_B2)
        ch_att = tf.nn.softmax(fc_att)

        res = tf.multiply(ch_att, trans_feat)
        return res
