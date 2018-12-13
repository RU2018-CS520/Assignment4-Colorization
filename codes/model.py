import theano
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, ReshapeLayer
from lasagne.nonlinearities import linear, tanh, sigmoid, rectify
from unpool_layer import Unpool2DLayer
from grayscale_layer import GrayscaleLayer

lr_policy = 'geometry'
lr_base = 0.0004
lr_step = 20
lr_coefficient = 0.8

def define_network_structure(input_var, **kwargs):
    # This function returns network, network output
    # Structure
    # Input - Grayscale - Conv1 + Max pooling - Dense - Dropout -
    # Unpool1 - Deconv1 - Output

    conv1_filter_cnt = 100
    conv1_filter_size = 5
    maxpool1_size = 2

    conv2_filter_cnt = 50
    conv2_filter_size = 5
    pool2_size = 2

    dense_units_cnt = 3000

    batch_size = input_var.shape[0]
    image_size = 32

    after_conv1 = image_size
    after_pool1 = (after_conv1 + maxpool1_size - 1) // maxpool1_size

    input = InputLayer(
        shape = (None, 3, image_size, image_size),
        input_var = input_var,
    )

    grayscale = GrayscaleLayer(incoming = input, )

    conv1 = Conv2DLayer(
        incoming = grayscale,
        num_filters = conv1_filter_cnt,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = tanh,
    )

    maxpool1 = MaxPool2DLayer(
        incoming = conv1,
        pool_size = maxpool1_size,
        stride = maxpool1_size,
    )

    dense1 = DenseLayer(
        incoming = maxpool1,
        num_units = dense_units_cnt,
        nonlinearity = tanh,
    )

    dense1 = DropoutLayer(
        incoming = dense1,
        p = 0.3,
    )

    pre_unpool1 = DenseLayer(
        incoming = dense1,
        num_units = conv1_filter_cnt * (after_pool1 ** 2),
        nonlinearity = tanh,
    )

    pre_unpool1 = ReshapeLayer(
        incoming = pre_unpool1,
        shape = (batch_size, conv1_filter_cnt) + (after_pool1, after_pool1),
    )

    unpool1 = Unpool2DLayer(
        incoming = pre_unpool1,
        kernel_size = maxpool1_size,
    )

    deconv1 = Conv2DLayer(
        incoming = unpool1,
        num_filters = 3,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = tanh,
    )

    output = ReshapeLayer(
        incoming = deconv1,
        shape = input_var.shape
    )

    return (output, lasagne.layers.get_output(output, deterministic = True))

def get_cost_updates(network, input_var, output, learning_rate, **kwargs):
    # Get cost
    params = lasagne.layers.get_all_params(network, trainable = True)

    batch_size = input_var.shape[0]
    flat_input = input_var.reshape((batch_size, 3072))
    flat_output= output.reshape((batch_size, 3072))

    flat_output = (flat_output + 1) / 2
    losses = theano.tensor.sum((flat_input - flat_output) ** 2, axis = 1)
    cost = theano.tensor.mean(losses)
    gradients = theano.tensor.grad(cost, params)

    updates = lasagne.updates.nesterov_momentum(gradients, params, learning_rate = learning_rate)

    return (cost, updates)
