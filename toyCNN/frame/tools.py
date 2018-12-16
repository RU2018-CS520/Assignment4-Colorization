import numpy as np

#3d pad
def pad(ix, iy, iz, padSize, iData):
	#in:
	#uint ix, iy, iz: iSize
	#uint padSize: outer pad size
	#np.ndarray iData with ndim = 3: data to be padded
	#out:
	#np.ndarray tData with ndim = 3: padded data

	if padSize == 0:
		return iData
	res = np.zeros((iz, ix+2*padSize, iy+2*padSize), dtype = np.float16)
	res[:, padSize: padSize+ix, padSize: padSize+iy] = iData
	return res

#4d pad
def pad4(ix, iy, iz, oz, padSize, iData):
	#uint ix, iy, iz: iSize
	#uint oz: output depth
	#uint padSize: outer pad size
	#np.ndarray iData with ndim = 3: data to be padded
	#out:
	#np.ndarray tData with ndim = 3: padded data

	if padSize == 0:
		return iData
	res = np.zeros((oz, iz, ix+2*padSize, iy+2*padSize), dtype = np.float16)
	res[:, :, padSize: padSize+ix, padSize: padSize+iy] = iData
	return res

#init numpy.random.seed
def setNpSeed(seed = 6983):
	#uint seed: random seed
	np.random.seed(seed)
	return


class lossFunc(object):
	'''
	def loss(o, y):
		#in:
		#np.ndarray o with ndim = 3: output data point
		#np.ndarray y with ndim = 3: expected output data point
		#out:
		#np.ndarray dl with ndim = 3: loss.derivative
		#np.ndarray loss with ndim = 3: loss for each node

		pass
	'''
	
	def norm2(o, y):
		dl = o - y
		return (dl, np.square(dl))