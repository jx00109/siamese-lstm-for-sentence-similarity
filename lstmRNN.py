# -*- coding:utf8 -*-
import tensorflow as tf


class LSTMRNN(object):
    def singleRNN(self, x, scope, cell='lstm', reuse=None):
        if cell == 'gru':
            with tf.variable_scope('grucell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.GRUCell(self.hidden_neural_size, reuse=tf.get_variable_scope().reuse)

        else:
            with tf.variable_scope('lstmcell' + scope, reuse=reuse, dtype=tf.float64):
                used_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_neural_size, forget_bias=1.0, state_is_tuple=True,
                                                         reuse=tf.get_variable_scope().reuse)

        with tf.variable_scope('cell_init_state' + scope, reuse=reuse, dtype=tf.float64):
            self.cell_init_state = used_cell.zero_state(self.batch_size, dtype=tf.float64)

        with tf.name_scope('RNN_' + scope), tf.variable_scope('RNN_' + scope, dtype=tf.float64):
            outs, _ = tf.nn.dynamic_rnn(used_cell, x, initial_state=self.cell_init_state, time_major=False,
                                        dtype=tf.float64)
        return outs

    def __init__(self, config, sess, is_training=True):
        self.keep_prob = config.keep_prob
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)

        num_step = config.num_step
        embed_dim = config.embed_dim
        self.input_data_s1 = tf.placeholder(tf.float64, [None, num_step, embed_dim])
        self.input_data_s2 = tf.placeholder(tf.float64, [None, num_step, embed_dim])
        self.target = tf.placeholder(tf.float64, [None])
        self.mask_s1 = tf.placeholder(tf.float64, [None, num_step])
        self.mask_s2 = tf.placeholder(tf.float64, [None, num_step])

        self.hidden_neural_size = config.hidden_neural_size
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)
        # with tf.name_scope('embedding_layer'):
        #     init_emb = tf.constant_initializer(table.g)
        #     embedding = tf.get_variable("embedding", shape=table.g.shape, initializer=init_emb, dtype=tf.float32)
        #     self.input_s1 = tf.nn.embedding_lookup(embedding, self.input_data_s1)
        #     self.input_s2 = tf.nn.embedding_lookup(embedding, self.input_data_s2)
        # 使用dropout会影响效果
        # if self.keep_prob < 1:
        #     self.input_s1 = tf.nn.dropout(self.input_s1, self.keep_prob)
        #     self.input_s2 = tf.nn.dropout(self.input_s2, self.keep_prob)

        with tf.name_scope('lstm_output_layer'):
            self.cell_outputs1 = self.singleRNN(x=self.input_data_s1, scope='side1', cell='lstm', reuse=None)
            self.cell_outputs2 = self.singleRNN(x=self.input_data_s2, scope='side1', cell='lstm', reuse=True)

        with tf.name_scope('Sentence_Layer'):
            # 此处得到句子向量，通过调整mask，可以改变句子向量的组成方式
            # 由于mask是用于指示句子的结束位置，所以此处使用sum函数而不是mean函数
            self.sent1 = tf.reduce_sum(self.cell_outputs1 * self.mask_s1[:, :, None], axis=1)
            self.sent2 = tf.reduce_sum(self.cell_outputs2 * self.mask_s2[:, :, None], axis=1)

        with tf.name_scope('loss'):
            diff = tf.abs(tf.subtract(self.sent1, self.sent2), name='err_l1')
            diff = tf.reduce_sum(diff, axis=1)
            self.sim = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
            self.loss = tf.square(tf.subtract(self.sim, tf.clip_by_value((self.target - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)))

        # with tf.name_scope('pearson_r'):
        #     _, self.pearson_r = tf.contrib.metrics.streaming_pearson_correlation(self.sim * 4.0 + 1.0, self.target)
        #     sess.run(tf.local_variables_initializer())

        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(self.loss)
            self.truecost = tf.reduce_mean(tf.square(tf.subtract(self.sim * 4.0 + 1.0, self.target)))

        # add summary
        cost_summary = tf.summary.scalar('cost_summary', self.cost)
        # r_summary = tf.summary.scalar('r_summary', self.pearson_r)
        mse_summary = tf.summary.scalar('mse_summary', self.truecost)

        if not is_training:
            return

        self.globle_step = tf.Variable(0, name="globle_step", trainable=False)
        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)

        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)
        self.summary = tf.summary.merge(
            [cost_summary, mse_summary, self.grad_summaries_merged])

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001, epsilon=1e-6)

        with tf.name_scope('train'):
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float64, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

    def assign_new_lr(self, session, lr_value):
        lr, _ = session.run([self.lr, self._lr_update], feed_dict={self.new_lr: lr_value})
        return lr

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})
