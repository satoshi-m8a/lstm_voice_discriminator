# -*- coding: utf-8 -*-
import tensorflow as tf

from data_loader import DataLoader
from model import Model


class Config(object):
    num_epochs = 1000
    batch_size = 1000
    num_steps = 300
    keep_prob = 0.8
    num_layers = 3
    hidden_size = 80
    max_grad_norm = 5
    decay_rate = 0.95
    learning_rate = 0.05


def main(_):
    config = Config()
    data_loader = DataLoader(config)
    model = Model(config, True)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for e in xrange(config.num_epochs):
            print "epoch {} / {}".format(e, config.num_epochs)
            sess.run(tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e)))
            state = model.initial_state.eval()
            for b in xrange(data_loader.num_batches):
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed)
                print train_loss


if __name__ == '__main__':
    tf.app.run()
