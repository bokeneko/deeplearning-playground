#!/opt/anaconda2/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf

import os
import os.path
import random
import struct
import time

random.seed(time.time())

COMMENT_MAX_LEN = 1000
CATCHCOPY_MAX_LEN = 64



class Seq2SeqGAN:
    def __init__(self, batch_size, embed_size=128, param_size=128, num_layer=64):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.param_size = param_size
        self.num_layer = num_layer

        self.filter_sizes = [2, 3, 4, 5, 6, 7]
        self.filter_num = 128

        self.encoder_inputs = tf.placeholder(tf.int32, [None, COMMENT_MAX_LEN])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, CATCHCOPY_MAX_LEN+1])
        self.targets = tf.placeholder(tf.int32, [None, CATCHCOPY_MAX_LEN])
        self.generator_inputs = tf.placeholder(tf.float32, [None, self.param_size])
        self.keep = tf.placeholder(tf.float32)

        self.create_encoder()
        self.create_decoder()
        self.create_generator()

    def create_encoder(self):
        with tf.variable_scope('embed'):
            embed_weight = tf.get_variable("embed_weight",
                                           shape=[0xffff, self.embed_size],
                                           initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                                           dtype=tf.float32)
            e = tf.nn.embedding_lookup(embed_weight, self.encoder_inputs)
            ex = tf.expand_dims(e, -1)
        with tf.variable_scope('encoder'):
            # conv
            with tf.variable_scope('conv') as scope:
                convs = []
                for filter_size in self.filter_sizes:
                    kernal = tf.get_variable("weight_{:02}".format(filter_size),
                                             shape=[filter_size, self.embed_size, 1, self.filter_num],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
                    b = tf.get_variable('b_{:02}'.format(filter_size), [1], tf.float32, tf.zeros_initializer)
                    conv = tf.nn.conv2d(ex, kernal, [1, 1, 1, 1], padding="VALID")
                    mean, variance = tf.nn.moments(conv, [0, 1, 2])
                    bn = tf.nn.relu(tf.nn.batch_normalization(conv, mean, variance, b, None, 1e-5))
                    conv1 = tf.nn.relu(bn, name="{}_{:02}".format(scope.name, filter_size))
                    conv2 = tf.nn.max_pool(conv1, [ 1, self.encoder_inputs.get_shape()[1] - filter_size + 1, 1, 1 ], [ 1, 1, 1, 1 ], 'VALID')
                    convs.append(conv2)
                conv = tf.concat(3, convs)
            # fully connected 1
            with tf.name_scope('fc1') as scope:
                total_filters = self.filter_num * len(self.filter_sizes)
                w = tf.get_variable("weight_fc1",
                                    shape=[total_filters, 512],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                h_fc1 = tf.matmul(tf.reshape(conv, [-1, total_filters]), w)
                b = tf.get_variable('b_fc1', [512], tf.float32, tf.zeros_initializer)
                mean, variance = tf.nn.moments(h_fc1, [0, 1])
                bn_fc1 = tf.nn.relu(tf.nn.batch_normalization(h_fc1, mean, variance, b, None, 1e-5))
            # dropout
            with tf.name_scope("dropout") as scope:
                h_dropout = tf.nn.dropout(bn_fc1, self.keep)
            # fully connected 2
            with tf.name_scope('fc2') as scope:
                w = tf.get_variable("weight_fc2",
                                    shape=[512, self.param_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
                h_fc2 = tf.matmul(h_dropout, w)
                b = tf.get_variable('b_fc2', [self.param_size], tf.float32, tf.zeros_initializer)
                mean, variance = tf.nn.moments(h_fc2, [0, 1])
                self.encoder_outputs = tf.nn.tanh(tf.nn.batch_normalization(h_fc2, mean, variance, b, None, 1e-5))


    def create_decoder(self):
        decoder_in = self.encoder_outputs
        with tf.variable_scope('decoder'):
            with tf.variable_scope('fc_reshape'):
                w = tf.get_variable('w_reshape', [self.param_size, 512], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                fc = tf.matmul(decoder_in, w)
                mean, variance = tf.nn.moments(fc, [0, 1])
                outputs = tf.nn.relu(tf.nn.batch_normalization(fc, mean, variance, None, None, 1e-5))
                outputs = tf.reshape(outputs, [self.batch_size, 4, 8, 16])
            # deconv1
            with tf.variable_scope('deconv1'):
                w = tf.get_variable('w_deconv1', [3, 3, 8, 16], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 8, 16, 8], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv2
            with tf.variable_scope('deconv2'):
                w = tf.get_variable('w_deconv2', [3, 3, 4, 8], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 16, 32, 4], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv3
            with tf.variable_scope('deconv3'):
                w = tf.get_variable('w_deconv3', [3, 3, 2, 4], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 32, 64, 2], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv4
            with tf.variable_scope('deconv4'):
                w = tf.get_variable('w_deconv4', [3, 3, 1, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 64, 128, 1], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            outputs = tf.reshape(outputs, [self.batch_size, 64, 128])

            # seq2seq
            with tf.variable_scope('lstm'):
                cell = tf.nn.rnn_cell.LSTMCell(
                                self.embed_size,
                                forget_bias=1.0,
                                use_peepholes=True,
                                state_is_tuple=True)
                cell = tf.nn.rnn_cell.MultiRNNCell(
                                [cell] * self.num_layer,
                                state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(
                                cell,
                                input_keep_prob=self.keep,
                                output_keep_prob=1.0)
                _, encoder_state = tf.nn.rnn(cell, tf.unpack(outputs, axis=1), dtype=tf.float32)

                cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, 0xffff)
                decoder_output, self.decoder_state = tf.nn.seq2seq.embedding_rnn_decoder(
                                tf.unpack(self.decoder_inputs, axis=1),
                                encoder_state,
                                cell,
                                0xffff,
                                self.embed_size,
                                feed_previous=False)
                decoder_output = tf.pack(decoder_output[:-1], axis=1)
        self.decoder_out = tf.argmax(decoder_output, 2)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(decoder_output, [-1, 0xffff])],
            [tf.expand_dims(tf.reshape(self.targets, [-1]), -1)],
            [tf.ones([self.batch_size * CATCHCOPY_MAX_LEN], dtype=tf.float32)])
        self.loss = tf.reduce_sum(loss) / self.batch_size


    def create_generator(self):
        decoder_in = self.generator_inputs
        with tf.variable_scope('decoder', reuse=True):
            with tf.variable_scope('fc_reshape'):
                w = tf.get_variable('w_reshape', [self.param_size, 512], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                fc = tf.matmul(decoder_in, w)
                mean, variance = tf.nn.moments(fc, [0, 1])
                outputs = tf.nn.relu(tf.nn.batch_normalization(fc, mean, variance, None, None, 1e-5))
                outputs = tf.reshape(outputs, [self.batch_size, 4, 8, 16])
            # deconv1
            with tf.variable_scope('deconv1'):
                w = tf.get_variable('w_deconv1', [3, 3, 8, 16], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 8, 16, 8], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv2
            with tf.variable_scope('deconv2'):
                w = tf.get_variable('w_deconv2', [3, 3, 4, 8], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 16, 32, 4], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv3
            with tf.variable_scope('deconv3'):
                w = tf.get_variable('w_deconv3', [3, 3, 2, 4], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 32, 64, 2], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            # deconv4
            with tf.variable_scope('deconv4'):
                w = tf.get_variable('w_deconv4', [3, 3, 1, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                dc = tf.nn.conv2d_transpose(outputs, w, [self.batch_size, 64, 128, 1], [1, 2, 2, 1])
                mean, variance = tf.nn.moments(dc, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, None, None, 1e-5))
            outputs = tf.reshape(outputs, [self.batch_size, 64, 128])

            # seq2seq
            with tf.variable_scope('lstm'):
                cell = tf.nn.rnn_cell.LSTMCell(
                                self.embed_size,
                                forget_bias=1.0,
                                use_peepholes=True,
                                state_is_tuple=True)
                cell = tf.nn.rnn_cell.MultiRNNCell(
                                [cell] * self.num_layer,
                                state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(
                                cell,
                                input_keep_prob=1.0,
                                output_keep_prob=1.0)
                _, encoder_state = tf.nn.rnn(cell, tf.unpack(outputs, axis=1), dtype=tf.float32)

                cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, 0xffff)
                decoder_output, _ = tf.nn.seq2seq.embedding_rnn_decoder(
                                tf.unpack(self.decoder_inputs, axis=1),
                                encoder_state,
                                cell,
                                0xffff,
                                self.embed_size,
                                feed_previous=True)
                decoder_output = tf.pack(decoder_output[:-1], axis=1)
        self.generator = tf.argmax(decoder_output, 2)


def main(data, batch_size=100, epoch_count=1000):
    model = Seq2SeqGAN(batch_size, num_layer=4, embed_size=128, param_size=128)
    train_op = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(model.loss)

    saver = tf.train.Saver(tf.all_variables())
    checkpoint_path = os.path.join("checkpoint_catchcopy", 'ckpt')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # restore
        if os.path.exists(checkpoint_path):
            print('restore variables:')
            saver.restore(sess, checkpoint_path)

        # setup for monitoring
        sample_z = sess.run(tf.random_uniform([model.batch_size, model.param_size], minval=-1.0, maxval=1.0))
        sample_dec_input = sess.run(tf.ones([model.batch_size, CATCHCOPY_MAX_LEN+1], dtype=tf.int32))

        # train
        for epoch in range(epoch_count):
            print "epoch: {}".format(epoch)
            random.shuffle(data)
            for batch_i in range(0, len(data), batch_size):
                ins_values = []
                dec_values = []
                target_values = []
                for r_id, comment, target, catchcopy in data[batch_i:batch_i+batch_size]:
                    ins_values.append(comment)
                    dec_values.append(catchcopy)
                    target_values.append(target)
                if len(ins_values) != batch_size:
                    continue
                start_time = time.time()
                feed_dict = {}
                feed_dict[model.encoder_inputs] = ins_values
                feed_dict[model.decoder_inputs] = dec_values
                feed_dict[model.targets] = target_values
                feed_dict[model.keep] = 0.8
                _, loss_value, decoder_out = sess.run([train_op, model.loss, model.decoder_out], feed_dict)
                duration = time.time() - start_time

                if sess.run(tf.is_nan(loss_value)):
                    raise Exception()

                # save generated images
                format_str = 'epoch {}, batch_count {}, loss = {:.8f} ({:.3f} sec/batch)'
                print(format_str.format(epoch, batch_i/batch_size, loss_value, duration))

                # save variables
                if batch_i % 10000 == 0:
                    filename = os.path.join("gen_catchcopy", '{:04}_{:06}.txt'.format(epoch, batch_i))
                    with open(filename, 'w') as f:
                        feed_dict = {}
                        feed_dict[model.generator_inputs] = sample_z
                        feed_dict[model.decoder_inputs] = sample_dec_input
                        feed_dict[model.keep] = 1.0
                        vs = sess.run(model.generator, feed_dict)
                        for v in vs:
                            s = u"".join([unichr(x) for x in v if x > 1]).encode("utf-8")
                            f.write("{}\n".format(s))
                    saver.save(sess, checkpoint_path)
        saver.save(sess, checkpoint_path)


if __name__ == "__main__":
    catchcopies = []
    with open("data/catchcopy.tsv") as f:
        for l in f:
            restaurant_id, catchcopy, comments = l.split("\t", 2)
            tmp_catchcopy = [ord(x) for x in catchcopy.decode("utf-8")]
            tmp_catchcopy = tmp_catchcopy[:CATCHCOPY_MAX_LEN]
            tmp_comment = [ord(x) for x in comments.decode("utf-8")]
            tmp_comment = tmp_comment[:COMMENT_MAX_LEN]
            if len(tmp_catchcopy) < 15:
                continue
            if len(tmp_comment) < 100:
                continue
            if len(tmp_catchcopy) < CATCHCOPY_MAX_LEN:
                tmp_catchcopy += ([0] * (CATCHCOPY_MAX_LEN - len(tmp_catchcopy)))
            if len(tmp_comment) < COMMENT_MAX_LEN:
                tmp_comment += ([0] * (COMMENT_MAX_LEN - len(tmp_comment)))
            catchcopies.append((restaurant_id, tmp_comment, tmp_catchcopy, [1] + tmp_catchcopy))
    main(catchcopies)
