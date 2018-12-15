import numpy as np
import timeit

import layer
from frame import funcs
from frame import tools

#use description to build a layer list
def buildNet(iSize, layerDescription, seedNum = None):
	#in:
	#tuple iSize = (ix, iy, iz): input size
	#list layerDescription = [(type, arg)]: config of layers
	#uint seedNum: init np.random
	#out:
	#list layerList = [layer.layer]: network layers

	#init random seed
	if seedNum is None:
		pass
	else:
		tools.setNpSeed(seedNum)

	#build layer
	tx, ty, tz = iSize
	layerList = []
	for layerType, layerArg in layerDescription:
		if layerType == 'c':
			kernelNum, padSize, stepLen, kernelSize = layerArg
			ox = (tx - kernelSize[0] + 2*padSize) // stepLen + 1
			oy = (ty - kernelSize[1] + 2*padSize) // stepLen + 1
			oz = kernelNum
			layerList.append(layer.conv((tx, ty, tz), (ox, oy, oz), kernelNum, padSize, stepLen, kernelSize))
		elif layerType == 'a':
			func, = layerArg
			ox = tx
			oy = ty
			oz = tz
			layerList.append(layer.active((tx, ty, tz), (ox, oy, oz), func))
		elif layerType == 'p':
			stepLen, func = layerArg
			ox = tx // stepLen
			oy = ty // stepLen
			oz = tz
			layerList.append(layer.pool((tx, ty, tz), (ox, oy, oz), stepLen, func))
		elif layerType == 'f':
			kernelNum, = layerArg
			ox = 1
			oy = 1
			oz = kernelNum
			layerList.append(layer.fullC((tx, ty, tz), (ox, oy, oz), kernelNum))
		else:
			print('E: network.buildNet: worng layerType')
		tx, ty, tz = ox, oy, oz
	return layerList



class net(object):
	def __init__(self, layerList, learnRate = 0.003, lossFunc = tools.lossFunc.norm2):
		#list layerList = [layer.layer]: network layers
		#float learnRate in [0: 1]: update step size
		#tools.lossFunc lossFunc: error functions
		self.layerList = layerList
		self.learnRate = learnRate
		self.lossFunc = lossFunc
		return

	def __repr__(self):
		description = 'net:'
		for layer in self.layerList:
			description = description + '\n' + repr(layer)
		return description
	def __str__(self):
		description = 'network.net:'
		for layer in self.layerList:
			description = description + '\n' + repr(layer)
		return description

	#forward input data
	def forward(self, x):
		#in:
		#np.ndarray x with ndim = 3: input data point
		#out:
		#np.ndarray x with ndim = 3: output data point
		tData = x
		for layer in self.layerList:
			tData = layer.forward(tData)
		return tData

	#use lossFunc get loss
	def getLoss(self, x, y):
		#in:
		#np.ndarray x with ndim = 3: input data point
		#np.ndarray y with ndim = 3: expected output data point
		#out:
		#np.ndarray dl with ndim = 3: loss.derivative
		#np.ndarray loss with ndim = 3: loss for each node
		oData = self.forward(x)
		dl, self.loss = self.lossFunc(oData, y)
		return dl, self.loss

	#backward modified loss
	def backward(self, loss):
		#in:
		#np.ndarray loss with ndim = 3: loss.derivative in previous layer
		#out:
		#np.ndarray loss with ndim = 3: loss.derivative in this layer
		for layer in reversed(self.layerList):
			loss = layer.backward(loss)
		return loss

	#use modified loss update kernel
	def update(self):
		for layer in self.layerList:
			layer.update(self.learnRate)
		return

	#iterate all data points and update
	def epoch(self, X, Y):
		#in:
		#list X = [x]: input data points
		#list Y = [y]: expected output data points
		#out:
		#float loss: average loss for all data points
		lossList = []
		for i in range(len(X)):
			dl, loss = self.getLoss(X[i], Y[i])
			lossList.append(loss)
			self.backward(dl)
			self.update()
		return np.mean(lossList)

	#use epoch to train network
	def train(self, tX, tY, vX, vY, maxIter, patience):
		#list tX = [x]: training input data points
		#list tY = [y]: training expected output data points
		#list vX = [x]: validation input data points
		#list vY = [y]: validation expected output data points
		#uint maxIter: max epoch numbers
		#float patience: early halt when converge

		maxPatience = patience
		bestLoss = float('inf')
		vLoss = float('inf')
		print('********Start training********')
		totalStart = timeit.default_timer()
		for i in range(maxIter):
			#get loss
			startTime = timeit.default_timer()
			loss = self.epoch(tX, tY)
			endTime = timeit.default_timer()
			print('epoch %i, mean loss: %.2f, for %.2fm' %(i, loss, (endTime - startTime)/60.))
			
			if loss < bestLoss:
				#validate model
				bestLoss = loss
				loss = self.test(vX, vY, validation = True)
				print('...validation loss: %.2f' %(loss))
				if loss < vLoss:
					vLoss = loss
					patience = min(maxPatience, patience + i/2.)
			else:
				#decrease patience
				patience = patience - i

			if patience < 0:
				#early halt
				print('no more patience.')
				break
		else:
			print('reach max iter.')

		totalEnd = timeit.default_timer()
		print('********Finish training********')
		print('best training loss: %.2f, best validation loss: %.2f, for %.2fm' %(bestLoss, vLoss, (totalEnd - totalStart)/60.))
		return

	#test performance
	def test(self, X, Y, validation = False):
		#in:
		#list X = [x]: test or validation input data points
		#list Y = [y]: test or validation expected output data points
		#bool validation: True: validation data; False: test data
		#out:
		#float loss: average loss for all data points
		loss = self.epoch(X, Y)
		if not validation:
			print('test loss: %.2f' %(loss))
		return loss

	#use network predict input
	def predict(self, X):
		#in:
		#list X = [x]: input data points
		#out:
		#list Y = [y]: output data points
		Y = []
		for x in X:
			Y.append(self.forward(x))
		return Y




if __name__ == '__main__':
	des = [('c', (2, 2, 2, (2,2))), ('p', (2, funcs.max())), ('a', (funcs.sigmoid(), )), ('f', (3, )), ('a', (funcs.leakyReLU(0.1), ))]
	ll = buildNet((4,4, 3), des, seedNum = 6983)
	nn = net(ll, 0.05)
	print(nn)
	iData = [np.random.rand(3, 4,4)]
	y = [0.1]
	oData = nn.train(iData, y, iData, y, 5, 100)
	# print(nn.predict(iData))
	# print(iData)
	# print(oData)
	# dl, loss = nn.getLoss(iData, 4)
	# print('')
	# # print(dl)
	# nn.backward(dl)
	# nn.update()
	# # print(nn.layerList[0].loss)
	# print('')
	# oData = nn.forward(iData)
	# print(oData)