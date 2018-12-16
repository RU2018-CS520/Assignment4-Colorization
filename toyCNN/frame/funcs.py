import numpy as np

'''
class func(object):
	#general func

	#original func
	def __call__(self, iData):
		#in:
		#np.ndarray iData with ndim = 3: input data point
		#out:
		#np.ndarray oData with ndim = 3: output data point

		pass
	
	#d(func) / d(iData)
	def derivative(self, iData, oData):
		#in:
		#np.ndarray iData with ndim = 3: input data point
		#np.ndarray oData with ndim = 3: output data point
		#out:
		#np.ndarray df with shape = iData.shape: d(func) / d(iData)

		pass
'''

#pool
class max(object):
	def __call__(self, iData):
		return np.max(iData, axis = (1, 2), keepdims = True)

	def derivative(self, iData, oData):
		res = np.zeros_like(iData, dtype = np.uint8)
		res[iData == oData] = 1
		return res

	def __repr__(self):
		return 'max'
	def __str__(self):
		return 'frame.actFunc: max'


class mean(object):
	def __call__(self, iData):
		return np.mean(iData, axis = (1, 2), keepdims = True)

	def derivative(self, iData, oData):
		res = np.full_like(iData, 1 / np.prod(iData.shape), dtype = np.float16)
		return res

	def __repr__(self):
		return 'mean'
	def __str__(self):
		return 'frame.actFunc: mean'


#active
class ReLU(object):
	def __call__(self, iData, kernel):
		return np.maximum(0, iData + kernel)

	def derivative(self, iData, oData, kernel):
		res = np.zeros_like(iData, dtype = np.uint8)
		res[(iData + kernel) == oData] = 1
		return res

	def __repr__(self):
		return 'ReLU'
	def __str__(self):
		return 'frame.actFunc: ReLU'


class leakyReLU(object):
	def __init__(self, factor = 0.1):
		self.factor = factor
		return

	def __call__(self, iData, kernel):
		tData = iData + kernel
		return np.maximum(self.factor * tData, tData)

	def derivative(self, iData, oData, kernel):
		tData = iData + kernel
		res = np.full_like(iData, self.factor, dtype = np.float16)
		res[tData == oData] = 1
		return res

	def __repr__(self):
		return 'leakyReLU(' + str(self.factor) +')'
	def __str__(self):
		return 'frame.actFunc: leakyReLU(' + str(self.factor) + ')'

		
class sigmoid(object):
	def __call__(self, iData, kernel):
		return 1 / (np.exp(-(iData + kernel)) + 1)

	def derivative(self, iData, oData, kernel = None):
		return oData * (1 - oData)

	def __repr__(self):
		return 'sigmoid'
	def __str__(self):
		return 'frame.actFunc: sigmoid'


class tanh(object):
	def __call__(self, iData, kernel):
		return 2 / (np.exp(-2 * (iData + kernel)) + 1) - 1

	def derivative(self, iData, oData, kernel = None):
		return 1 - np.square(oData)

	def __repr__(self):
		return 'tanh'
	def __str__(self):
		return 'frame.actFunc: tanh'


#conv and fullC
class weight(object):
	def __call__(self, iData, kernel):
		return np.sum(iData * kernel, axis = 1)

	def derivative(self, iData, oData, kernel):
		return 1

	def __repr__(self):
		return 'weight'
	def __str__(self):
		return 'frame.actFunc: weight'