import os
import PIL
import random
import theano
import numpy as np

def save_image(arr, file_name = 'arr.png', path = 'pics'):
    # Save arr from numpy.ndarray
    arr = np.transpose(arr, (1, 2, 0))
    img = PIL.Image.fromarray(arr)

    if not os.path.exists(path):
        os.makedirs(path)
    img.save(path + '/' + file_name)

def get_grayscale_images(images, color = False):
    # Get grayscale of images using LUMA coding
    # images: 4D tensor or np.ndarray (index, channel, height, width)
    r = 0.299
    g = 0.587
    b = 0.114

    len = images.shape
    h = images.shape[2]
    w = images.shape[3]

    ret_images = r * images[0:len, 0, 0:h, 0:w] + g * images[0:len, 0, 0:h, 0:w]+ b * images[0:len, 0, 0:h, 0:w]

    if color == True:
        if isinstance(images, np.ndarray):
            ret_images = np.concatenate((ret_images, ret_images, ret_images), axis = 1)
        else:
            ret_images = theano.tensor.concatenate((ret_images, ret_images, ret_images), axis = 1)
    return ret_images

def save_model(network, iteration, model_name, learning_rate = 0.0, path = 'models'):
    # Save our model to pickle file
    params = layers.get_all_param_values(network)
    file_name = model_name + "-iter" + str(iteration) + ".pickle"
    file_path = path + '/' + file_name
    print("Saving model to: %s" % file_path)

    if not os.path.exists(path):
        os.makedirs(path)
    with open(file_path, 'wb') as file:
        pickle.dump(obj = {'params': params, 'iteration': iteration, 'learning_rate': learning_rate, }, file = file, protocol = -1)

def load_model(network, file_name, path = 'models'):
    # Load model from pickle file
    file_path = path + '/' + file_name
    print("Loading model from: %s" % file_path)

    with open(file_path, 'rb') as file:
        dict = pickle.load(file)
        layers.set_all_param_values(network, dict[b'params'])
    return {'iteration': dict[b'iteration'], 'learning_rate': dict[b'learning_rate']}

def plot_sample(images, mapping, model_name, iteration, images_per_row = 5, path = 'pics'):
    # Plot images to file
    # image can be np.ndarray or PIL.Image
    # mapping is a theano function which produces outputs from images
    file_path = path + '/' + model_name
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if not isinstance(images, np.ndarray):
        images = np.array(images.eval())

    len = images.shape[0]
    height = images.shape[2]
    width = images.shape[3]

    gray_images = get_grayscale_images(images, color = True)
    output_images = mapping(images)
    sample_images = np.array([]).reshape((0,) + images.shape[1:])

    for i in range(len):
        sample_images = np.concatenate([sample_images, images[i:i+1]], axis = 0)
        sample_images = np.concatenate([sample_images, gray_images[i:i+1]], axis = 0)
        sample_images = np.concatenate([sample_images, output_images[i:i+1]], axis = 0)

    sample_images = reshape_images_for_output((R, G, B, None), image_shape = (height, width), tile_shape = (len // images_per_row, 3 * images_per_row), tile_gap = (1, 1))

    img = Image.fromarray(sample_images)
    img_name = ('Iteration %d.png' % iteration)
    img.save(file_path + '/' + img_name)

def reshape_images_for_output(images, image_shape, tile_shape, tile_gap = (1, 1)):
    # Reshape images to human-friendly format
    # iterate each element in three lists
    output_shape = [(i + k) * j - k for i, j, k in zip(image_shape, tile_shape, tile_gap)]

    if isinstance(images, tuple):

