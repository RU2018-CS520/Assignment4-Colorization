import frame
import theano
import lasagne

class GrayscaleLayer(lasagne.layers.Layer):
    # Transfer input to grayscale
    def __init__(self, incoming, **kwargs):
        super(GrayscaleLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = 1
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return frame.get_grayscale_images(input)
