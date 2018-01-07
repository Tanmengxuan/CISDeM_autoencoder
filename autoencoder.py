#insert new 1
#insert new 2
#cuc edit
#cuc 2nd edit
#remove conflict
import tensorflow as tf
import numpy as np
import math
from sklearn import preprocessing
from inits import *
import time
import matplotlib.pyplot as plt
from os.path import isfile

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('input_dim', 80, 'Number of dimensions in the input to the network.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('train_percentage', 0.9, 'for training and val')
flags.DEFINE_integer('batch_size', 100, '#samples in mini batch')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('data_normalize', 0, 'normalize data or not')

# 1 Layers
flags.DEFINE_integer('ael1_hidden1', 80, 'Number of units in hidden layer 1.')

# 2 Layers
flags.DEFINE_integer('ael2_hidden1', 15, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('ael2_hidden2', 8, 'Number of units in hidden layer 2.')

# 3 Layers
flags.DEFINE_integer('ael3_hidden1', 15, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('ael3_hidden2', 10, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('ael3_hidden3', 8, 'Number of units in hidden layer 3.')

# 4 Layers
flags.DEFINE_integer('ael4_hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('ael4_hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('ael4_hidden3', 32, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('ael4_hidden4', 16, 'Number of units in hidden layer 4.')

x = tf.placeholder(tf.float32, [None, FLAGS.input_dim])
keep_prob = tf.placeholder(tf.float32)


def train_auto_encoder_1l(data, load_pretrained=False):
    if FLAGS.data_normalize:
        data = preprocessing.normalize(data, norm='l2')

    num_sample, num_dim = data.shape

    num_train = int(math.floor(num_sample * FLAGS.train_percentage))
    data_shuffle = np.random.permutation(data)
    train_data = data_shuffle[0:num_train]
    val_data = data_shuffle[num_train:]
    print('val_data.shape', val_data.shape)
    print('data is ', train_data.shape)

    # x = tf.placeholder(tf.float32, [None, num_dim])
    #
    # keep_prob = tf.placeholder(tf.float32)

    weight_enc1 = glorot([num_dim, FLAGS.ael1_hidden1], name='weight_enc1')
    bias_enc1 = tf.Variable(tf.random_normal([FLAGS.ael1_hidden1], mean=0.0, stddev=0.01), name='bias_enc1')

    weight_dec1 = glorot([FLAGS.ael1_hidden1, num_dim])
    bias_dec1 = tf.Variable(tf.random_normal([num_dim], mean=0.0, stddev=0.01))

    x_drop = tf.nn.dropout(x, keep_prob)

    enc1 = tf.matmul(x_drop, weight_enc1)
    enc1 = tf.nn.tanh(enc1 + bias_enc1)
    enc1 = tf.nn.dropout(enc1, keep_prob, name="enc1")

    # dec1 = tf.nn.dropout(enc1, keep_prob)
    dec1 = tf.matmul(enc1, weight_dec1)
    dec1 = tf.nn.tanh(dec1 + bias_dec1, name="dec1")

    # The loss function:
    loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(dec1, x), 2.0), 1))
    # Addition of weight decay:
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_enc1) + tf.nn.l2_loss(bias_enc1))
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_dec1) + tf.nn.l2_loss(bias_dec1))

    # Optimize:
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    optimizer = opt.minimize(loss)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_train = []
    cost_val = []

    num_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
    model_filename = "models/autoencoder_{}f_2l_{}_{}.ckpt".format(FLAGS.input_dim,
                                                                    FLAGS.ael2_hidden1,
                                                                    FLAGS.ael2_hidden2)
    if load_pretrained:
        saver.restore(sess, model_filename)
    else:
        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Training step
            train_data_shuffle = np.random.permutation(train_data)
            for i in range(num_batch):
                if i == num_batch - 1:
                    features = train_data_shuffle[FLAGS.batch_size * (num_batch - 1):]
                else:
                    features = train_data_shuffle[FLAGS.batch_size * i: FLAGS.batch_size * (i + 1)]
                outs = sess.run([optimizer, loss], feed_dict={x: features, keep_prob: 1 - FLAGS.dropout})
                cost_train.append(outs[1])
            # Validation
            outs_val = sess.run([loss], feed_dict={x: val_data, keep_prob: 1})
            cost_val.append(outs_val[0])
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(cost_train[-(num_batch + 1):-1])),
                  "val_loss=", "{:.5f}".format(outs_val[0]), "time=", "{:.5f}".format(time.time() - t))
            plt.plot(range(len(cost_train)), cost_train)
            plt.savefig('train_loss_ae.png')
            # plt.show()
            plt.close()
            plt.plot(range(len(cost_val)), cost_val)
            plt.savefig('val_loss_ae.png')
            # plt.show()
            plt.close()

            # Early stopping condition:
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        saver.save(sess, model_filename)
        sess.close()

    return enc1, dec1


def train_auto_encoder_2l(data, load_pretrained=False):
    if FLAGS.data_normalize:
        data = preprocessing.normalize(data, norm='l2')

    num_sample, num_dim = data.shape

    num_train = int(math.floor(num_sample * FLAGS.train_percentage))
    data_shuffle = np.random.permutation(data)
    train_data = data_shuffle[0:num_train]
    val_data = data_shuffle[num_train:]
    print('val_data.shape', val_data.shape)
    print('data is ', train_data.shape)

    # x = tf.placeholder(tf.float32, [None, num_dim])
    #
    # keep_prob = tf.placeholder(tf.float32)

    weight_enc1 = glorot([num_dim, FLAGS.ael2_hidden1], name='weight_enc1')
   # weight_enc1 = tf.get_variable("weight_enc1",[num_dim,80], initializer= tf.contrib.layers.xavier_initializer(seed=1))                   
    bias_enc1 = tf.Variable(tf.random_normal([FLAGS.ael2_hidden1], mean=0.0, stddev=0.01), name='bias_enc1')
   # bias_enc1 = tf.get_variable("bias_enc1",[80,1], initializer = tf.zeros_initializer())
    weight_enc2 = glorot([FLAGS.ael2_hidden1, FLAGS.ael2_hidden2], name='weight_enc2')
    bias_enc2 = tf.Variable(tf.random_normal([FLAGS.ael2_hidden2], mean=0.0, stddev=0.01), name='bias_enc2')

    weight_dec1 = glorot([FLAGS.ael2_hidden1, num_dim])
    bias_dec1 = tf.Variable(tf.random_normal([num_dim], mean=0.0, stddev=0.01))
    weight_dec2 = glorot([FLAGS.ael2_hidden2, FLAGS.ael2_hidden1])
    bias_dec2 = tf.Variable(tf.random_normal([FLAGS.ael2_hidden1], mean=0.0, stddev=0.01))

    x_drop = tf.nn.dropout(x, keep_prob)

    enc1 = tf.matmul(x_drop, weight_enc1)
    enc1 = tf.nn.tanh(enc1 + bias_enc1)
    enc1 = tf.nn.dropout(enc1, keep_prob, name="enc1")

    enc2 = tf.matmul(enc1, weight_enc2)
    enc2 = tf.nn.relu(enc2 + bias_enc2, name="enc2")

    # dec2 = tf.nn.dropout(enc2, keep_prob)
    dec2 = tf.matmul(enc2, weight_dec2)
    dec2 = tf.nn.relu(dec2 + bias_dec2, name="dec2")

    dec1 = tf.nn.dropout(dec2, keep_prob)
    dec1 = tf.matmul(dec1, weight_dec1)
    dec1 = tf.nn.tanh(dec1 + bias_dec1, name="dec1")

    # The loss function:
    loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(dec1, x), 2.0), 1))
    # Addition of weight decay:
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_enc1) + tf.nn.l2_loss(bias_enc1) +
                                  tf.nn.l2_loss(weight_enc2) + tf.nn.l2_loss(bias_enc2))
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_dec1) + tf.nn.l2_loss(bias_dec1) +
                                  tf.nn.l2_loss(weight_dec2) + tf.nn.l2_loss(bias_dec2))

    # Optimize:
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    optimizer = opt.minimize(loss)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_train = []
    cost_val = []

    num_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
    model_filename = "models/autoencoder_{}f_2l_{}_{}.ckpt".format(FLAGS.input_dim,
                                                                    FLAGS.ael2_hidden1,
                                                                    FLAGS.ael2_hidden2)
    if load_pretrained:
        saver.restore(sess, model_filename)
    else:
        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Training step
            train_data_shuffle = np.random.permutation(train_data)
            for i in range(num_batch):
                if i == num_batch - 1:
                    features = train_data_shuffle[FLAGS.batch_size * (num_batch - 1):]
                else:
                    features = train_data_shuffle[FLAGS.batch_size * i: FLAGS.batch_size * (i + 1)]
                outs = sess.run([optimizer, loss], feed_dict={x: features, keep_prob: 1 - FLAGS.dropout})
                cost_train.append(outs[1])
            # Validation
            outs_val = sess.run([loss], feed_dict={x: val_data, keep_prob: 1})
            cost_val.append(outs_val[0])
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(cost_train[-(num_batch + 1):-1])),
                  "val_loss=", "{:.5f}".format(outs_val[0]), "time=", "{:.5f}".format(time.time() - t))
            plt.plot(range(len(cost_train)), cost_train)
            plt.savefig('train_loss_ae.png')
            # plt.show()
            plt.close()
            plt.plot(range(len(cost_val)), cost_val)
            plt.savefig('val_loss_ae.png')
            # plt.show()
            plt.close()

            # Early stopping condition:
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        saver.save(sess, model_filename)
        sess.close()

    return enc2, dec1


def train_auto_encoder_3l(data, load_pretrained=False):
    if FLAGS.data_normalize:
        data = preprocessing.normalize(data, norm='l2')

    num_sample, num_dim = data.shape

    num_train = int(math.floor(num_sample * FLAGS.train_percentage))
    data_shuffle = np.random.permutation(data)
    train_data = data_shuffle[0:num_train]
    val_data = data_shuffle[num_train:]
    print('val_data.shape', val_data.shape)
    print('data is ', train_data.shape)

    # x = tf.placeholder(tf.float32, [None, num_dim])
    #
    # keep_prob = tf.placeholder(tf.float32)

    weight_enc1 = glorot([num_dim, FLAGS.ael3_hidden1], name='weight_enc1')
    bias_enc1 = tf.Variable(tf.random_normal([FLAGS.ael3_hidden1], mean=0.0, stddev=0.01), name='bias_enc1')
    weight_enc2 = glorot([FLAGS.ael3_hidden1, FLAGS.ael3_hidden2], name='weight_enc2')
    bias_enc2 = tf.Variable(tf.random_normal([FLAGS.ael3_hidden2], mean=0.0, stddev=0.01), name='bias_enc2')
    weight_enc3 = glorot([FLAGS.ael3_hidden2, FLAGS.ael3_hidden3], name='weight_enc3')
    bias_enc3 = tf.Variable(tf.random_normal([FLAGS.ael3_hidden3], mean=0.0, stddev=0.01), name='bias_enc3')

    weight_dec1 = glorot([FLAGS.ael3_hidden1, num_dim])
    bias_dec1 = tf.Variable(tf.random_normal([num_dim], mean=0.0, stddev=0.01))
    weight_dec2 = glorot([FLAGS.ael3_hidden2, FLAGS.ael3_hidden1])
    bias_dec2 = tf.Variable(tf.random_normal([FLAGS.ael3_hidden1], mean=0.0, stddev=0.01))
    weight_dec3 = glorot([FLAGS.ael3_hidden3, FLAGS.ael3_hidden2])
    bias_dec3 = tf.Variable(tf.random_normal([FLAGS.ael3_hidden2], mean=0.0, stddev=0.01))

    x_drop = tf.nn.dropout(x, keep_prob)

    enc1 = tf.matmul(x_drop, weight_enc1)
    enc1 = tf.nn.tanh(enc1 + bias_enc1)
    enc1 = tf.nn.dropout(enc1, keep_prob)

    enc2 = tf.matmul(enc1, weight_enc2)
    enc2 = tf.nn.relu(enc2 + bias_enc2)
    enc2 = tf.nn.dropout(enc2, keep_prob)

    enc3 = tf.matmul(enc2, weight_enc3)
    enc3 = tf.nn.relu(enc3 + bias_enc3)

    dec3 = tf.nn.dropout(enc3, keep_prob)
    dec3 = tf.matmul(dec3, weight_dec3)
    dec3 = tf.nn.relu(dec3 + bias_dec3)

    dec2 = tf.nn.dropout(dec3, keep_prob)
    dec2 = tf.matmul(dec2, weight_dec2)
    dec2 = tf.nn.relu(dec2 + bias_dec2)

    dec1 = tf.nn.dropout(dec2, keep_prob)
    dec1 = tf.matmul(dec1, weight_dec1)
    dec1 = tf.nn.tanh(dec1 + bias_dec1)

    # The loss function:
    loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(dec1, x), 2.0), 1))
    # Addition of weight decay:
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_enc1) + tf.nn.l2_loss(bias_enc1) +
                                  tf.nn.l2_loss(weight_enc2) + tf.nn.l2_loss(bias_enc2) +
                                  tf.nn.l2_loss(weight_enc3) + tf.nn.l2_loss(bias_enc3))
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_dec1) + tf.nn.l2_loss(bias_dec1) +
                                  tf.nn.l2_loss(weight_dec2) + tf.nn.l2_loss(bias_dec2) +
                                  tf.nn.l2_loss(weight_dec3) + tf.nn.l2_loss(bias_dec3))

    # Optimize:
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    optimizer = opt.minimize(loss)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_train = []
    cost_val = []

    num_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
    model_filename = "models/autoencoder_{}f_3l_{}_{}_{}.ckpt".format(FLAGS.input_dim,
                                                                    FLAGS.ael3_hidden1,
                                                                    FLAGS.ael3_hidden2,
                                                                    FLAGS.ael3_hidden3)
    if load_pretrained:
        saver.restore(sess, model_filename)
    else:
        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Training step
            train_data_shuffle = np.random.permutation(train_data)
            for i in range(num_batch):
                if i == num_batch - 1:
                    features = train_data_shuffle[FLAGS.batch_size * (num_batch - 1):]
                else:
                    features = train_data_shuffle[FLAGS.batch_size * i: FLAGS.batch_size * (i + 1)]
                outs = sess.run([optimizer, loss], feed_dict={x: features, keep_prob: 1 - FLAGS.dropout})
                cost_train.append(outs[1])
            # Validation
            outs_val = sess.run([loss], feed_dict={x: val_data, keep_prob: 1})
            cost_val.append(outs_val[0])
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(cost_train[-(num_batch + 1):-1])),
                  "val_loss=", "{:.5f}".format(outs_val[0]), "time=", "{:.5f}".format(time.time() - t))
            plt.plot(range(len(cost_train)), cost_train)
            plt.savefig('train_loss_ae.png')
            # plt.show()
            plt.close()
            plt.plot(range(len(cost_val)), cost_val)
            plt.savefig('val_loss_ae.png')
            # plt.show()
            plt.close()

            # Early stopping condition:
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        saver.save(sess, model_filename)
        sess.close()

    return enc3, dec1


def train_auto_encoder_4l(data, load_pretrained=False):
    if FLAGS.data_normalize:
        data = preprocessing.normalize(data, norm='l2')

    num_sample, num_dim = data.shape

    num_train = int(math.floor(num_sample * FLAGS.train_percentage))
    data_shuffle = np.random.permutation(data)
    train_data = data_shuffle[0:num_train]
    val_data = data_shuffle[num_train:]
    print('val_data.shape', val_data.shape)
    print('data is ', train_data.shape)

    # x = tf.placeholder(tf.float32, [None, num_dim])
    #
    # keep_prob = tf.placeholder(tf.float32)

    weight_enc1 = glorot([num_dim, FLAGS.ael4_hidden1], name='weight_enc1')
    bias_enc1 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden1], mean=0.0, stddev=0.01), name='bias_enc1')
    weight_enc2 = glorot([FLAGS.ael4_hidden1, FLAGS.ael4_hidden2], name='weight_enc2')
    bias_enc2 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden2], mean=0.0, stddev=0.01), name='bias_enc2')
    weight_enc3 = glorot([FLAGS.ael4_hidden2, FLAGS.ael4_hidden3], name='weight_enc3')
    bias_enc3 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden3], mean=0.0, stddev=0.01), name='bias_enc3')
    weight_enc4 = glorot([FLAGS.ael4_hidden3, FLAGS.ael4_hidden4], name='weight_enc4')
    bias_enc4 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden4], mean=0.0, stddev=0.01), name='bias_enc4')

    weight_dec1 = glorot([FLAGS.ael4_hidden1, num_dim])
    bias_dec1 = tf.Variable(tf.random_normal([num_dim], mean=0.0, stddev=0.01))
    weight_dec2 = glorot([FLAGS.ael4_hidden2, FLAGS.ael4_hidden1])
    bias_dec2 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden1], mean=0.0, stddev=0.01))
    weight_dec3 = glorot([FLAGS.ael4_hidden3, FLAGS.ael4_hidden2])
    bias_dec3 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden2], mean=0.0, stddev=0.01))
    weight_dec4 = glorot([FLAGS.ael4_hidden4, FLAGS.ael4_hidden3])
    bias_dec4 = tf.Variable(tf.random_normal([FLAGS.ael4_hidden3], mean=0.0, stddev=0.01))

    x_drop = tf.nn.dropout(x, keep_prob)

    enc1 = tf.matmul(x_drop, weight_enc1)
    enc1 = tf.nn.tanh(enc1 + bias_enc1)
    enc1 = tf.nn.dropout(enc1, keep_prob)

    enc2 = tf.matmul(enc1, weight_enc2)
    enc2 = tf.nn.relu(enc2 + bias_enc2)
    enc2 = tf.nn.dropout(enc2, keep_prob)

    enc3 = tf.matmul(enc2, weight_enc3)
    enc3 = tf.nn.relu(enc3 + bias_enc3)
    enc3 = tf.nn.dropout(enc3, keep_prob)

    enc4 = tf.matmul(enc3, weight_enc4)
    enc4 = tf.nn.relu(enc4 + bias_enc4)

    dec4 = tf.nn.dropout(enc4, keep_prob)
    dec4 = tf.matmul(dec4, weight_dec4)
    dec4 = tf.nn.relu(dec4 + bias_dec4)

    dec3 = tf.nn.dropout(dec4, keep_prob)
    dec3 = tf.matmul(dec3, weight_dec3)
    dec3 = tf.nn.relu(dec3 + bias_dec3)

    dec2 = tf.nn.dropout(dec3, keep_prob)
    dec2 = tf.matmul(dec2, weight_dec2)
    dec2 = tf.nn.relu(dec2 + bias_dec2)

    dec1 = tf.nn.dropout(dec2, keep_prob)
    dec1 = tf.matmul(dec1, weight_dec1)
    dec1 = tf.nn.tanh(dec1 + bias_dec1)

    # The loss function:
    loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(dec1, x), 2.0), 1))
    # Addition of weight decay:
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_enc1) + tf.nn.l2_loss(bias_enc1) +
                                  tf.nn.l2_loss(weight_enc2) + tf.nn.l2_loss(bias_enc2) +
                                  tf.nn.l2_loss(weight_enc3) + tf.nn.l2_loss(bias_enc3) +
                                  tf.nn.l2_loss(weight_enc4) + tf.nn.l2_loss(bias_enc4))
    loss += FLAGS.weight_decay * (tf.nn.l2_loss(weight_dec1) + tf.nn.l2_loss(bias_dec1) +
                                  tf.nn.l2_loss(weight_dec2) + tf.nn.l2_loss(bias_dec2) +
                                  tf.nn.l2_loss(weight_dec3) + tf.nn.l2_loss(bias_dec3) +
                                  tf.nn.l2_loss(weight_dec4) + tf.nn.l2_loss(bias_dec4))

    # Optimize:
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    optimizer = opt.minimize(loss)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    cost_train = []
    cost_val = []

    num_batch = int(math.ceil(float(num_train) / FLAGS.batch_size))
    model_filename = "models/autoencoder_{}f_4l_{}_{}_{}_{}.ckpt".format(FLAGS.input_dim,
                                                               FLAGS.ael4_hidden1,
                                                                FLAGS.ael4_hidden2,
                                                                FLAGS.ael4_hidden3,
                                                                FLAGS.ael4_hidden4)
    if load_pretrained:
        saver.restore(sess, model_filename)
    else:
        # Train model
        for epoch in range(FLAGS.epochs):

            t = time.time()
            # Training step
            train_data_shuffle = np.random.permutation(train_data)
            for i in range(num_batch):
                if i == num_batch - 1:
                    features = train_data_shuffle[FLAGS.batch_size * (num_batch - 1):]
                else:
                    features = train_data_shuffle[FLAGS.batch_size * i: FLAGS.batch_size * (i + 1)]
                outs = sess.run([optimizer, loss], feed_dict={x: features, keep_prob: 1 - FLAGS.dropout})
                cost_train.append(outs[1])
            # Validation
            outs_val = sess.run([loss], feed_dict={x: val_data, keep_prob: 1})
            cost_val.append(outs_val[0])
            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(np.mean(cost_train[-(num_batch + 1):-1])),
                  "val_loss=", "{:.5f}".format(outs_val[0]), "time=", "{:.5f}".format(time.time() - t))
            plt.plot(range(len(cost_train)), cost_train)
            plt.savefig('train_loss_ae.png')
            # plt.show()
            plt.close()
            plt.plot(range(len(cost_val)), cost_val)
            plt.savefig('val_loss_ae.png')
            # plt.show()
            plt.close()

            # Early stopping condition:
            if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")

        saver.save(sess, model_filename)
        sess.close()

    return enc4, dec1


def train_auto_encoder(data, load_pre_trained=False):
    return train_auto_encoder_4l(data, load_pre_trained)


def encode_data(encoder, decoder, data):
    # num_samples, num_dim = data.shape
    # x = tf.placeholder(tf.float32, [None, num_dim])
    # keep_prob = tf.placeholder(tf.float32)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    outs = sess.run([encoder], feed_dict={x: data, keep_prob: 1})
    reduced_x = outs[0]
    outs = sess.run([decoder], feed_dict={x: data, keep_prob: 1})
    reconstructed_x = outs[0]
   # distance = np.subtract(reduced_x, reconstructed_x)
    distance = None
    print("Reduced dimension from {} to {}.".format(data.shape[1], reduced_x.shape[1]))
    sess.close()

    return reduced_x, reconstructed_x, distance
