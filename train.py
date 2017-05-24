# -*- coding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import time
from lstmRNN import LSTMRNN
import data_helper
from gensim.models.word2vec import KeyedVectors
from scipy.stats import pearsonr

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'the batch_size of the training procedure')
flags.DEFINE_float('lr', 0.0001, 'the learning rate')
flags.DEFINE_float('lr_decay', 0.95, 'the learning rate decay')
flags.DEFINE_integer('emdedding_dim', 300, 'embedding dim')
flags.DEFINE_integer('hidden_neural_size', 50, 'LSTM hidden neural size')
flags.DEFINE_integer('max_len', 73, 'max_len of training sentence')
flags.DEFINE_integer('valid_num', 100, 'epoch num of validation')
flags.DEFINE_integer('checkpoint_num', 1000, 'epoch num of checkpoint')
flags.DEFINE_float('init_scale', 0.1, 'init scale')

flags.DEFINE_float('keep_prob', 0.5, 'dropout rate')
flags.DEFINE_integer('num_epoch', 360, 'num epoch')
flags.DEFINE_integer('max_decay_epoch', 100, 'num epoch')
flags.DEFINE_integer('max_grad_norm', 5, 'max_grad_norm')
flags.DEFINE_string('out_dir', os.path.abspath(os.path.join(os.path.curdir, "runs_new1")), 'output directory')
flags.DEFINE_integer('check_point_every', 20, 'checkpoint every num epoch ')


class Config(object):
    hidden_neural_size = FLAGS.hidden_neural_size
    embed_dim = FLAGS.emdedding_dim
    keep_prob = FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm = FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num = FLAGS.valid_num
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every


def cut_data(data, rate):
    x1, x2, y, mask_x1, mask_x2 = data

    n_samples = len(x1)

    # 打散数据集
    sidx = np.random.permutation(n_samples)

    ntrain = int(np.round(n_samples * (1.0 - rate)))

    train_x1 = [x1[s] for s in sidx[:ntrain]]
    train_x2 = [x2[s] for s in sidx[:ntrain]]
    train_y = [y[s] for s in sidx[:ntrain]]
    train_m1 = [mask_x1[s] for s in sidx[:ntrain]]
    train_m2 = [mask_x2[s] for s in sidx[:ntrain]]

    valid_x1 = [x1[s] for s in sidx[ntrain:]]
    valid_x2 = [x2[s] for s in sidx[ntrain:]]
    valid_y = [y[s] for s in sidx[ntrain:]]
    valid_m1 = [mask_x1[s] for s in sidx[ntrain:]]
    valid_m2 = [mask_x2[s] for s in sidx[ntrain:]]

    # 打散划分好的训练和测试集
    train_data = [train_x1, train_x2, train_y, train_m1, train_m2]
    valid_data = [valid_x1, valid_x2, valid_y, valid_m1, valid_m2]

    return train_data, valid_data


def evaluate(model, session, data, global_steps=None, summary_writer=None):
    x1, x2, y, mask_x1, mask_x2 = data

    fetches = [model.truecost, model.sim, model.target]
    feed_dict = {}
    feed_dict[model.input_data_s1] = x1
    feed_dict[model.input_data_s2] = x2
    feed_dict[model.target] = y
    feed_dict[model.mask_s1] = mask_x1
    feed_dict[model.mask_s2] = mask_x2
    model.assign_new_batch_size(session, len(x1))
    cost, sim, target = session.run(fetches, feed_dict)

    pearson_r = pearsonr(sim, target)

    dev_summary = tf.summary.scalar('dev_pearson_r', pearson_r)

    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary, global_steps)
        summary_writer.flush()
    return cost, pearson_r


def run_epoch(model, session, data, global_steps, valid_model, valid_data, train_summary_writer,
              valid_summary_writer=None):
    for step, (s1, s2, y, mask_s1, mask_s2) in enumerate(data_helper.batch_iter(data, batch_size=FLAGS.batch_size)):

        feed_dict = {}
        feed_dict[model.input_data_s1] = s1
        feed_dict[model.input_data_s2] = s2
        feed_dict[model.target] = y
        feed_dict[model.mask_s1] = mask_s1
        feed_dict[model.mask_s2] = mask_s2
        model.assign_new_batch_size(session, len(s1))
        fetches = [model.truecost, model.sim, model.target, model.train_op, model.summary]
        cost, sim, target, _, summary = session.run(fetches, feed_dict)

        pearson_r = pearsonr(sim, target)

        train_summary_writer.add_summary(summary, global_steps)
        train_summary_writer.flush()

        if (global_steps % 100 == 0):
            valid_cost, valid_pearson_r = evaluate(valid_model, session, valid_data, global_steps,
                                                   valid_summary_writer)
            print(
                "the %i step, train cost is: %f and the train pearson_r is %f and the valid cost is %f the valid pearson_r is %f" % (
                    global_steps, cost, pearson_r, valid_cost, valid_pearson_r))

        global_steps += 1

    return global_steps


def train_step():
    config = Config()
    eval_config = Config()
    eval_config.keep_prob = 1.0

    # gpu_config=tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session() as session:
        # 这个初始化不好，效果极差
        initializer = tf.random_normal_initializer(0.0, 0.2, dtype=tf.float32)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = LSTMRNN(config=config, sess=session, is_training=True)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            valid_model = LSTMRNN(config=eval_config, sess=session, is_training=False)
            test_model = LSTMRNN(config=eval_config, sess=session, is_training=False)

        # add summary
        train_summary_dir = os.path.join(config.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

        dev_summary_dir = os.path.join(eval_config.out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

        # add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        tf.global_variables_initializer().run()

        global_steps = 1
        begin_time = int(time.time())

        print("loading the dataset...")
        pretrained_word_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',
                                                                  binary=True)

        pre_train_data = data_helper.load_data(FLAGS.max_len, pretrained_word_model, datapath='./data/stsallrmf.p',
                                               embed_dim=FLAGS.emdedding_dim)
        data = data_helper.load_data(FLAGS.max_len, pretrained_word_model, datapath='./data/semtrain.p',
                                     embed_dim=FLAGS.emdedding_dim)
        test_data = data_helper.load_data(FLAGS.max_len, pretrained_word_model, datapath='./data/semtest.p',
                                          embed_dim=FLAGS.emdedding_dim)

        print("length of pre-train set:", len(pre_train_data[0]))
        print("length of train set:", len(data[0]))
        print("length of test set:", len(test_data[0]))
        print("begin pre-training")

        for i in range(70):
            print("the %d epoch pre-training..." % (i + 1))
            lr = model.assign_new_lr(session, config.lr)
            print("current learning rate is %f" % lr)

            # 11000+ data
            train_data, valid_data = cut_data(pre_train_data, 0.05)

            global_steps = run_epoch(model, session, train_data, global_steps, valid_model, valid_data,
                                     train_summary_writer, dev_summary_writer)

        path = saver.save(session, checkpoint_prefix, global_steps)
        print("pre-train finish.")
        print("Saved pre-train model chechpoint to{}\n".format(path))
        print("begin training")

        for i in range(config.num_epoch):
            print("the %d epoch training..." % (i + 1))
            # lr_decay = config.lr_decay ** max(i - config.max_decay_epoch, 0.0)
            lr = model.assign_new_lr(session, config.lr)
            print('current learning rate is %f' % lr)

            train_data, valid_data = cut_data(data, 0.1)

            global_steps = run_epoch(model, session, train_data, global_steps, valid_model, valid_data,
                                     train_summary_writer, dev_summary_writer)

            if i % config.checkpoint_every == 0:
                path = saver.save(session, checkpoint_prefix, global_steps)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time = int(time.time())
        print("training takes %d seconds already\n" % (end_time - begin_time))
        test_cost, test_pearson_r = evaluate(test_model, session, test_data)
        print("the test data cost is %f" % test_cost)
        print("the test data pearson_r is %f" % test_pearson_r)

        print("program end!")


def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()
