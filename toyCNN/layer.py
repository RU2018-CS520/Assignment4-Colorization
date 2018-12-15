import numpy as np

from frame import tools
from frame import funcs

class layer(object):
	#general layer frame, DO NOT use it!
	def __init__(self, iSize, oSize):
		#tuple iSize = (ix, iy, iz): input shape
		#tuple oSize = (ox, oy ,oz): output shape

		self.ix, self.iy, self.iz = iSize
		self.ox, self.oy, self.oz = oSize
		self.type = ''
		return

	def __repr__(self):
		return self.type + ': ' + str((self.ix, self.iy, self.iz)) + ' -> ' + str((self.ox, self.oy, self.oz))
	def __str__(self):
		return 'layer: ' + self.type + ' layer: ' + str((self.ix, self.iy, self.iz)) + ' -> ' + str((self.ox, self.oy, self.oz))

	#check if oSize matches iSize and other configs
	def validCheck(self):
		pass

	#forward input data
	def forward(self, iData):
		#in:
		#np.ndarray x with ndim = 3: input data point
		#out:
		#np.ndarray x with ndim = 3: output data point
		pass

	#backward modified loss
	def backward(self, loss):
		#in:
		#np.ndarray loss with ndim = 3: loss.derivative in previous layer
		#out:
		#np.ndarray loss with ndim = 3: loss.derivative in this layer
		pass

	#use modified loss update kernel
	def update(self, learnRate):
		#float learnRate in [0: 1]: update step size
		pass


class conv(layer):
	#convolutional layer
	def __init__(self, iSize, oSize, kernelNum, padSize, stepLen, kernelSize):
		#uint kernelNum: number of kernels
		#uint padSize: outer pad size
		#uint stepLen: kernel moving step size
		#tuple kernelSize = (x, y): kernel shape

		super(conv, self).__init__(iSize, oSize)
		self.type = 'c'

		self.kernelNum = kernelNum
		self.padSize = padSize
		self.stepLen = stepLen
		self.kernelSize = kernelSize
		self.func = funcs.weight()

		self.validCheck()

		#init conv kernel
		self.kernel = np.random.rand(self.kernelNum , self.iz, self.kernelSize[0], self.kernelSize[1]).astype(np.float16)
		return


	def validCheck(self):
		tx = (self.ix - self.kernelSize[0] + 2*self.padSize) // self.stepLen + 1
		ty = (self.iy - self.kernelSize[1] + 2*self.padSize) // self.stepLen + 1
		tz = self.kernelNum
		if self.ox != tx or self.oy != ty or self.oz != tz:
			print('E: layer.conv: wrong conv parameters')
			exit()
		return


	def forward(self, iData):
		self.iData = iData
		#pad iData
		tData = tools.pad(self.ix, self.iy, self.iz, self.padSize, iData)

		#init oData
		res = np.empty((self.oz, self.ox, self.oy), dtype = np.float16)

		#convolution
		for row in range(self.ox):
			t = row * self.stepLen
			b = t + self.kernelSize[0]
			for col in range(self.oy):
				l = col * self.stepLen
				r = l + self.kernelSize[1]

				res[:, row, col] = np.sum(self.func(tData[:, t: b, l: r], self.kernel), axis = (1, 2))

		self.oData = res
		return self.oData


	def backward(self, loss):
		tz = self.iz
		tx = self.ix+2*self.padSize
		ty = self.iy+2*self.padSize

		#init oLoss
		res = np.empty((tz, self.oz, tx, ty), dtype = np.float16)

		#transpose iAxis and oAxis
		#transpose oAxis to the last axis to match iLoss
		rKernel = np.transpose(self.kernel, (1, 2, 3, 0))

		#de-convolution
		for row in range(self.ox):
			t = row * self.stepLen
			b = t + self.kernelSize[0]
			for col in range(self.oy):
				l = col * self.stepLen
				r = l + self.kernelSize[1]

				#transpose oAxis back
				res[:, :, t: b, l: r] = np.transpose((loss[:, row, col] * rKernel), (0, 3, 1, 2))

		#de-pad
		if self.padSize == 0:
			self.loss = res
		else:
			self.loss = res[:, :, self.padSize: self.padSize+self.ix, self.padSize: self.padSize+self.iy]

		#sum oAxis to backward
		res = np.sum(self.loss, axis = 1)
		return res


	def update(self, learnRate):
		#pad iData and loss
		tData = tools.pad(self.ix, self.iy, self.iz, self.padSize, self.iData)
		loss = tools.pad4(self.ix, self.iy, self.iz, self.oz, self.padSize, np.transpose(self.loss, (1, 0, 2, 3)))

		#stack convlution
		res = np.zeros_like(self.kernel, dtype = np.float16)
		for row in range(self.ox):
			t = row * self.stepLen
			b = t + self.kernelSize[0]
			for col in range(self.oy):
				l = col * self.stepLen
				r = l + self.kernelSize[1]

				res = res + tData[:, t: b, l: r] * loss[:, :, t: b, l: r]

		#average
		res = res / (self.ox * self.oy)

		#update
		self.kernel = self.kernel - learnRate * res
		return



class active(layer):
	#activation layer
	def __init__(self, iSize, oSize, func):
		#funcs func: active function

		super(active, self).__init__(iSize, oSize)
		self.type = 'a'
		self.func = func

		self.validCheck()

		#init bias
		self.kernel = np.random.rand(self.iz, self.ix, self.iy).astype(np.float16)
		return


	def validCheck(self):
		tx = self.ix
		ty = self.iy
		tz = self.iz
		if self.ox != tx or self.oy != ty or self.oz != tz:
			print('E: layer.active: wrong active parameters')
			exit()
		return


	def forward(self, iData):
		self.iData = iData
		self.oData = self.func(iData, self.kernel)
		return self.oData


	def backward(self, loss):
		res = loss * self.func.derivative(self.iData, self.oData, self.kernel)

		#backward w = 1, bias w = bias
		self.loss = self.kernel * res
		return res


	def update(self, learnRate):

		#bias o = 1
		self.kernel = self.kernel - learnRate * self.loss
		return


class pool(layer):
	def __init__(self, iSize, oSize, stepLen, func):
		#uint stepLen: kernel moving step size
		#funcs func: pool function

		super(pool, self).__init__(iSize, oSize)
		self.type = 'p'
		self.stepLen = stepLen
		self.kernelSize = (stepLen, stepLen)
		self.func = func

		self.validCheck()
		return


	def validCheck(self):
		tx = self.ix // self.stepLen
		ty = self.iy // self.stepLen
		tz = self.iz
		if self.ox != tx or self.oy != ty or self.oz != tz:
			print('E: frame.layer.pool: wrong pool parameters')
			exit()
		return


	def forward(self, iData):
		self.iData = iData

		#convolution
		res = np.empty((self.oz, self.ox, self.oy), dtype = np.float16)
		for row in range(self.ox):
			t = row * self.stepLen
			b = t + self.kernelSize[0]
			for col in range(self.oy):
				l = col * self.stepLen
				r = l + self.kernelSize[1]

				res[:, row, col] = np.sum(self.func(iData[:, t: b, l: r]), axis = (1, 2))

		self.oData = res
		return self.oData


	def backward(self, loss):
		res = np.empty((self.iz, self.ix, self.iy), dtype = np.float16)

		#de-convolution
		for row in range(self.ox):
			t = row * self.stepLen
			b = t + self.kernelSize[0]
			for col in range(self.oy):
				l = col * self.stepLen
				r = l + self.kernelSize[1]

				res[:, t: b, l: r] = loss[row, col] * self.func.derivative(self.iData[:, row, col], self.oData[:, row, col])

		return res


	def update(self, learnRate):
		#nothing to do
		return



class fullC(layer):
	def __init__(self, iSize, oSize, kernelNum):
		#uint kernelNum: number of outputs

		super(fullC, self).__init__(iSize, oSize)
		self.type = 'f'
		self.kernelNum = kernelNum
		self.func = funcs.weight()

		self.validCheck()

		self.kernel = np.random.rand(self.kernelNum, self.iz, self.ix, self.iy).astype(np.float16)
		return


	def validCheck(self):
		tx = 1
		ty = 1
		tz = self.kernelNum
		if self.ox != tx or self.oy != ty or self.oz != tz:
			print('E: frame.layer.fullC: wrong active parameters')
			exit()
		return


	def forward(self, iData):
		self.iData = iData
		self.oData = self.func(iData, self.kernel)
		return self.oData


	def backward(self, loss):

		#transpose iAxis and oAxis
		rKernel = np.transpose(self.kernel, (1, 0, 2, 3))
		self.loss = loss * rKernel

		#sum oAxis to backward
		res = np.sum(self.loss, axis = 1)
		return res

	def update(self, learnRate):

		#transpose oAxis and iAxis to match iData
		self.kernel = self.kernel - learnRate * self.loss.transpose(1,0,2,3) * self.iData
		return