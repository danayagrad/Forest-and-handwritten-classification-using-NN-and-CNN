import tensorflow as tf
import numpy as np
from datetime import datetime


#initialises neurons in layer and performs activation function on input
def neuron_layer(X, n_neurons, name, lambdaReg, activation=None):
    #X: inputs, n_neurons: outputs
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2.0 / np.sqrt(n_inputs + n_neurons)
        #initilaise weights and biases
        W = tf.get_variable("weights", shape=(n_inputs, n_neurons),
                            initializer=tf.truncated_normal_initializer(stddev=stddev),regularizer=tf.contrib.layers.l2_regularizer(lambdaReg))
        b = tf.Variable(tf.zeros([n_neurons]),name="bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z



def heavy_side(z, name=None):
    return tf.nn.relu(tf.math.sign(z), name=name)

def leaky_relu(z, name=None):
    return tf.maximum(0.2*z,z, name=name)


def conv2d(input_data, num_input_channels, num_filters, filter_shape, stride, pool_shape, name):

    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    n_inputs = int(input_data.get_shape()[1])
    stddev = 2.0 / np.sqrt(n_inputs + num_filters)

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=stddev),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [stride, stride, 1, 1], padding='SAME')

    # add the bias
    out_layer = tf.nn.bias_add(out_layer, bias)

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = int(np.ceil(len(X) / batch_size))
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch