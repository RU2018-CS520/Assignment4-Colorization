import os
import sys
import time
import frame
import theano
import lasagne
import numpy as np
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, ReshapeLayer
from lasagne.nonlinearities import linear, tanh, sigmoid, rectify
from unpool_layer import Unpool2DLayer
from grayscale_layer import GrayscaleLayer

lr = 0.0002
lr_step = 15
lr_coefficient = 0.8
lr_mode = 'geometric'

def build_cnn(input_var = None):
    # This function returns network, network prediction
    # Structure
    # Input - Grayscale - Conv1 + Max pooling - Dense - Dropout -
    # Unpool1 - Deconv1 - Output

    conv1_filter_cnt = 100
    conv1_filter_size = 5
    maxpool1_size = 2

    conv2_filter_cnt = 50
    conv2_filter_size = 5
    maxpool2_size = 2

    dense_units_cnt = 3000

    batch_size = input_var.shape[0]
    image_size = 32

    after_conv1 = image_size
    after_pool1 = (after_conv1 + maxpool1_size - 1) // maxpool1_size

    l_in = InputLayer(
        shape = (None, 3, image_size, image_size),
        input_var = input_var,
    )

    l_in_grayscale = GrayscaleLayer(incoming = l_in, )

    print(lasagne.layers.get_output_shape(l_in_grayscale))

    l_conv1 = Conv2DLayer(
        incoming = l_in_grayscale,
        num_filters = conv1_filter_cnt,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = tanh,
    )
    print(lasagne.layers.get_output_shape(l_conv1))

    l_maxpool1 = MaxPool2DLayer(
        incoming = l_conv1,
        pool_size = maxpool1_size,
        stride = maxpool1_size,
    )
    print(lasagne.layers.get_output_shape(l_maxpool1))

    l_dense1 = DenseLayer(
        incoming = l_maxpool1,
        num_units = dense_units_cnt,
        nonlinearity = tanh,
    )
    print(lasagne.layers.get_output_shape(l_dense1))

    l_drop1 = DropoutLayer(
        incoming = l_dense1,
        p = 0.3,
    )
    print(lasagne.layers.get_output_shape(l_drop1))

    l_pre_unpool1 = DenseLayer(
        incoming = l_drop1,
        num_units = conv1_filter_cnt * (after_pool1 ** 2),
        nonlinearity = tanh,
    )
    print(lasagne.layers.get_output_shape(l_pre_unpool1))

    #l_drop2 = DropoutLayer(
    #    incoming = l_pre_unpool1,
    #    p = 0.3,
    #)

    l_pre_unpool1 = ReshapeLayer(
        incoming = l_pre_unpool1,
        shape = (batch_size, conv1_filter_cnt) + (after_pool1, after_pool1),
    )
    print(lasagne.layers.get_output_shape(l_pre_unpool1))

    l_unpool1 = Unpool2DLayer(
        incoming = l_pre_unpool1,
        kernel_size = maxpool1_size,
    )
    print(lasagne.layers.get_output_shape(l_unpool1))

    l_deconv1 = Conv2DLayer(
        incoming = l_unpool1,
        num_filters = 3,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = tanh,
    )
    print(lasagne.layers.get_output_shape(l_deconv1))

    l_out = ReshapeLayer(
        incoming = l_deconv1,
        shape = input_var.shape
    )
    print(lasagne.layers.get_output_shape(l_out))

    return (l_out, lasagne.layers.get_output(l_out, deterministic = True))

def get_loss_and_updates(network, input_var, prediction, learning_rate, **kwargs):
    # Get loss and updates 
    params = lasagne.layers.get_all_params(network, trainable = True)

    batch_size = input_var.shape[0]
    input = input_var.reshape((batch_size, 3072))
    output = prediction.reshape((batch_size, 3072))

    output = (output + 1) / 2
    loss = theano.tensor.sum((input - output) ** 2, axis = 1)
    #regularization = theano.tensor.sum(theano.tensor.std(theano.tensor.reshape(output, (batch_size, 3, 1024)), axis = 1), axis = 1)
    #loss += 0.01 * regularization
    loss = theano.tensor.mean(loss)
    gradients = theano.tensor.grad(loss, params)

    updates = lasagne.updates.nesterov_momentum(gradients, params, learning_rate = learning_rate)

    return (loss, updates)

def train(batch_size = 100, max_iterations = 500, load_model_name = None, load_model_path = ''):
    # Load training dataset
    train_dataset_x = frame.load_train_dataset()[0].astype(theano.config.floatX)
    train_dataset_x = theano.shared(train_dataset_x, borrow = True)

    validation_dataset_x = frame.load_validation_dataset()[0].astype(theano.config.floatX)
    validation_dataset_x = theano.shared(validation_dataset_x, borrow = True)

    train_batches_size = train_dataset_x.get_value(borrow = True).shape[0] / batch_size
    validation_batches_size = validation_dataset_x.get_value(borrow = True).shape[0] / batch_size

    print('Total number of train batches: ' + repr(train_batches_size))
    print('Total number of validation batches: ' + repr(validation_batches_size))

    input_var = theano.tensor.tensor4('X')
    target_var = theano.tensor.tensor4('Y')
    index = theano.tensor.lscalar('index')

    # Build network
    network, prediction = build_cnn(input_var)

    # Create mapping
    mapping = theano.function(inputs = [input_var], outputs = prediction)

    # Learning rate setting
    learning_rate = theano.shared(np.float32(lr))

    print('Learning rate mode: ' + lr_mode)
    print('Leraning rate: ' + repr(lr))

    if lr_mode == 'geometric':
        print('lr_step = ' + repr(lr_step))
        print('lr_coefficient = ' + repr(lr_coefficient))

    # Loss expression for training
    #loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #loss = loss.mean()

    # Update expression for training
    #params = lasagne.layers.get_all_params(network, trainable = True)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = learning_rate, momentum = 0.9)

    # Functions for training
    #train_costs, train_updates = model.get_cost_updates(network = network, input_var = input_var, output = prediction, learning_rate = learning_rate, )
    #f = theano.function([index], train_costs, updates = train_updates, givens = {input_var:train_dataset_x[index * batch_size:(index + 1) * batch_size]})
    #f = theano.function([input_var, target_var], loss, updates = updates)
    loss, updates = get_loss_and_updates(network, input_var, prediction, learning_rate, )
    f = theano.function([index], loss, updates = updates, givens = {input_var:train_dataset_x[index * batch_size:(index + 1) * batch_size]})

    # Loss expression for validation
    loss_validation = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss_validation = loss_validation.mean()

    # Functions for validation
    val_loss, val_updates = get_loss_and_updates(network, input_var, prediction, learning_rate, )
    f_validate = theano.function([index], val_loss, givens = {input_var:validation_dataset_x[index * batch_size:(index + 1) * batch_size]})

    # Try to load previous trained model
    if load_model_name == '':
        iteration_start = 0
    else:
        res = frame.load_model(network = network, file_name = load_model_name, path = load_model_path, )
        iteration_start = res['iteration'] + 1

    # Logging
    if not os.path.exists('logs'):
        os.makedirs('logs')

    file_log_train = open('logs/log_train.csv', 'w')
    file_log_validation = open('logs/log_val.csv', 'w')

    # Training
    print('Started training')

    try:
        for iteration in range(iteration_start, max_iterations):
            start_time = time.time()
            loss_history = []
            for batch_index in range(int(train_batches_size)):
                loss_history.append(f(batch_index))

            print(file_log_train, 'Iteration: %d, mean loss = %f' % (iteration, np.mean(loss_history)))
            file_log_train.flush()
            print ('Training iteration %d took %.0fs, learning rate = %f, mean loss=%f' % (iteration, time.time() - start_time, learning_rate.eval(), np.mean(loss_history)))

            # Validation
            if iteration % 10 == 1:
                loss_history = []
                for batch_index in range(int(validation_batches_size)):
                    loss_history.append(f_validate(batch_index))

                print('Validating...')
                print(file_log_validation, 'Iteration: %d, mean loss = %f' % (iteration, np.mean(loss_history)))
                file_log_validation.flush()
                print('Mean validation loss = %f' % np.mean(loss_history))

            # Adjust learning rate
            if lr_mode == 'geometric':
                if iteration % lr_step == 1 and iteration > iteration_start + 1:
                    learning_rate *= lr_coefficient

            # Automatically save model to file every 50 iterations
            if iteration % 50 == 1:
                frame.save_model(network = network, iteration = iteration, model_name = 'cnn', learning_rate = learning_rate.eval(), path = 'models',)

            # Plot sample and filter every 10 iterations
            if iteration % 10 == 1:
                frame.plot_sample(
                    images = validation_dataset_x[:500],
                    mapping = mapping,
                    model_name = 'cnn',
                    iteration = iteration,
                    images_per_row = 5,
                    path = 'pics',
                )
                frame.plot_filter(
                    filters = lasagne.layers.get_all_param_values(network)[0],
                    model_name = 'cnn',
                    iteration = iteration,
                    repeat = 10,
                    filter_per_row = 8,
                    path = 'pics',
                )
            # Reset time
            start_time = time.time()

    except KeyboardInterrupt:
        if input('\nSave the model ? (y / n) ') == 'y':
            frame.save_model(
                network = network,
                iteration = iteration,
                model_name = 'cnn',
                learning_rate = learning_rate.eval(),
                path = 'models',
            )
        exit()

if __name__ == '__main__':
    train(batch_size = 100, load_model_name = 'cnn-iter51.pickle', load_model_path = '/common/users/sl1560/temp')
    #train(batch_size = 100, load_model_name = '', load_model_path = '')
