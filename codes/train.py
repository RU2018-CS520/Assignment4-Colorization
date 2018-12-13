import sys
import frame
import theano
import lasagne
import importlib
import numpy as np

network_name = sys.argv[1]
network_structure = importlib.machinery.SourceFileLoader(network_structure, '%s.py' % network_name)

def train(batch_size = 100, max_iterations = 500, save_iter = 100, info_iter = 10, val_iter = 10, log_path = 'logs', info_path = 'pics', save_path = 'models', load_path = 'models', load_model_name = None):
    # Load training dataset
    train_dataset_x = frame.load_train_dataset()[0].astype(theano.config.floatX)
    train_dataset_x = theano.shared(train_dataset_x, borrow = True)

    validation_dataset_x = frame.load_validation_dataset[0].astype(theano.config.floatX)
    validation_dataset_x = theano.shared(validation_dataset_x, borrow = True)

    train_batches_size = train_dataset_x.get_value(borrow = True).shape[0] / batch_size
    validation_batches_size = validation_batches_size.get_value(borrow = True).shape[0] / batch_size

    print('Total number of train batches' + repr(train_batches_size))
    print('Total number of validation batches' + repr(validation_batches_size))

    input_var = theano.tensor.tensor4('X')
    output_var = theano.tensor.tensor4('Y')
    index = theano.tensor.lscalar('index')

    # Build network
    network = network_structure.define
