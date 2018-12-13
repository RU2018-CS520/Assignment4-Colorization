import theano
import lasagne

class Unpool2DLayer(lasagne.layers.Layer):
    # Unpool a layer over its last 2D
    def __init__(self, incoming, kernel_size, nonlinearity = lasagne.nonlinearities.linear, **kwargs):
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[2] = input_shape[2] * self.kernel_size
        output_shape[3] = input_shape[3] * self.kernel_size
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return self.nonlinearity(input.repeat(self.kernel_size, axis = 2).repeat(self.kernel_size, axis = 3))
