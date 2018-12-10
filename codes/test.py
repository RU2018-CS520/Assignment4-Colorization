import numpy

test = numpy.arange(50*3*32*32).reshape((50, 3, 32, 32))
print(test.shape)
print(test[:, 0, :, :].shape)
