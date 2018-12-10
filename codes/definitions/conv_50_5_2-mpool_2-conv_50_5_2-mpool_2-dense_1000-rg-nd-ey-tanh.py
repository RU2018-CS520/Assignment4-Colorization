#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:33:41 2018

@author: zx123
"""

import theano.tensor as T
import lasagne
import lasagne.layers as layers
import our_layers


def define_model(input_var, **kwargs):
    """ 
    Defines the model and returns (network, validation network output)
        
    -Return layers.get_output(final_layer_name) if validation network output and 
        train network output are the same
    
    -For example, return layers.get_output(final_layer_name, deterministic = true) 
        if there is a dropout layer
            
    -Use **kwargs to pass model specific parameters
    """
    
    conv1_filter_count = 50
    conv1_filter_size = 5
    pool1_size = 2
    
    conv2_filter_count = 50
    conv2_filter_size = 5
    pool2_size = 2
    
    n_dense_units = 1000
    
    batch_size = input_var.shape[0]
    image_size = 32
    # note: ignore_border is True by defualt, so rounding down
    after_pool1 = image_size // pool1_size
    after_pool2 = after_pool1 // pool2_size
    
    input = layers.InputLayer(
        shape = (None, 3, image_size, image_size),
        input_var = input_var
    )
    
    greyscale_input = our_layers.GreyscaleLayer(
        incoming = input,
        random_greyscale = True,
    )
    
    conv1 = layers.Conv2DLayer(
        incoming = greyscale_input,
        num_filters = conv1_filter_count,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    pool1 = layers.MaxPool2DLayer(
        incoming = conv1,
        pool_size = pool1_size,
        stride = pool1_size,
    )
    
    conv2 = layers.Conv2DLayer(
        incoming = pool1,
        num_filters = conv2_filter_count,
        filter_size = conv2_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    pool2 = layers.MaxPool2DLayer(
        incoming = conv2,
        pool_size = pool2_size,
        stride = pool2_size,
    )
    
    dense1 = layers.DenseLayer(
        incoming = pool2,
        num_units = n_dense_units, 
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    pre_unpool2 = layers.DenseLayer(
        incoming = dense1,
        num_units = conv2_filter_count * (after_pool2 ** 2),
        nonlinearity = lasagne.nonlinearities.tanh,
    )

    pre_unpool2 = layers.ReshapeLayer(
        incoming = pre_unpool2, 
        shape = (batch_size, conv2_filter_count) + (after_pool2, after_pool2),
    )
    
    unpool2 = our_layers.Unpool2DLayer(
        incoming = pre_unpool2,
        kernel_size = pool2_size,
    )

    deconv2 = layers.Conv2DLayer(
        incoming = unpool2,
        num_filters = conv1_filter_count,
        filter_size = conv2_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    unpool1 = our_layers.Unpool2DLayer(
        incoming = deconv2,
        kernel_size = pool1_size,
    )

    deconv1 = layers.Conv2DLayer(
        incoming = unpool1,
        num_filters = 3,
        filter_size = conv1_filter_size,
        stride = 1,
        pad = 'same',
        nonlinearity = lasagne.nonlinearities.tanh,
    )
  
    output = layers.ReshapeLayer(
        incoming = deconv1,
        shape = input_var.shape
    )
    
    return output

def get_cost_updates(network, input_var, output, learning_rate, **kwargs):
    """ 
    Defines and returns cost and updates of the network
    -output can be different from layers.get_output(network), because train 
        and validation networks may differ from each other
    
    -Use **kwargs to pass model specific parameters
    """
    
    params = layers.get_all_params(network, trainable = True)

    batch_size = input_var.shape[0]
    flat_input = input_var.reshape((batch_size, 3072))
    flat_output= output.reshape((batch_size, 3072))

    # scale flat_output to [0, 1]
    flat_output = (flat_output + 1) / 2

    # cross entrophy loss
    # losses = -T.sum(flat_input * T.log(flat_output) + (1 - flat_input) * T.log(1 - flat_output), axis = 1) 
     
    # euclidean loss
    losses = T.sum((flat_input - flat_output) ** 2, axis = 1)
    
    # add saturation loss
    # saturation = -T.sum(T.std(T.reshape(flat_output, (batch_size, 3, 1024)), axis = 1), axis = 1)
    # losses = losses + 0.2 * saturation

    cost = T.mean(losses)
    print(losses)
    print(cost)
    print(params)
    gradients = T.grad(cost, params)

    # stochastic gradient descent
    # updates = [(param, param - learning_rate * gradient) for param, gradient in zip(params, gradients)]
    
    # rmsprop
    # updates = lasagne.updates.rmsprop(gradients, params, learning_rate = learning_rate)
    
    # momentum
    updates = lasagne.updates.nesterov_momentum(gradients, params, learning_rate = learning_rate) 
    
    return (cost, updates)
