import i2a
import frame
import train
import random
import theano
import lasagne
import numpy as np

cnt = 20
batch_size = 20
load_model_name = 'cnn-iter51.pickle'
learning_rate = 0.0002

input_var = theano.tensor.tensor4('X')
target_var = theano.tensor.tensor4('Y')
index = theano.tensor.lscalar('index')

# Build network
network, prediction = train.build_cnn(input_var)


# Create mapping
mapping = theano.function(inputs = [input_var], outputs = prediction)

# Load model
model = frame.load_model(network = network, file_name = load_model_name, path = '/common/users/sl1560/temp', )
learning_rate = model['learning_rate']
iteration = model['iteration']
params = lasagne.layers.get_all_params(network, trainable = True)

# Load validation data
validation_dataset_x = i2a.batch('data/', 1, 20)[0].astype(theano.config.floatX)
#validation_dataset_x = frame.load_validation_dataset()[0].astype(theano.config.floatX)
validation_dataset_x = theano.shared(validation_dataset_x, borrow = True)
validation_dataset_size = validation_dataset_x.get_value(borrow = True).shape[0]
validation_batches_size = 1

# Functions for validation
val_loss, val_updates = train.get_loss_and_updates(network, input_var, prediction, learning_rate, )
f_validate = theano.function([index], val_loss, givens = {input_var:validation_dataset_x[index * batch_size:(index + 1) * batch_size]})

# Validation
print('Started validation')
loss_history = []
for batch_index in range(int(validation_batches_size)):
    print(batch_index)
    loss_history.append(f_validate(batch_index))

print('Validating...')
print('Mean validation loss = %f' % np.mean(loss_history))

validaion_sample = np.array(validation_dataset_x.eval())
np.random.shuffle(validaion_sample)

frame.plot_sample(
    images = validaion_sample[0:cnt],
    mapping = mapping,
    model_name = 'experiment',
    iteration = iteration,
    images_per_row = 5,
    path = 'pics',
)
frame.plot_filter(
    filters = lasagne.layers.get_all_param_values(network)[0],
    model_name = 'cnn-51',
    iteration = iteration,
    repeat = 10,
    filter_per_row = 8,
    path = 'pics',
)
