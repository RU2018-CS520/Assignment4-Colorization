import numpy as np
import timeit

import layer
from frame import funcs
from frame import tools

def buildNet(iSize, layerDescription, seedNum = None):
	if seedNum is None:
		pass
	else:
		tools.setNpSeed(seedNum)

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


	def forward(self, x):
		tData = x
		for layer in self.layerList:
			tData = layer.forward(tData)
		return tData

	def getLoss(self, x, y):
		oData = self.forward(x)
		dl, self.loss = self.lossFunc(oData, y)
		return dl, self.loss

	def backward(self, loss):
		for layer in reversed(self.layerList):
			loss = layer.backward(loss)
		return loss

	def update(self):
		for layer in self.layerList:
			layer.update(self.learnRate)
		return


	def epoch(self, X, Y):
		lossList = []
		for i in range(X):
			dl, loss = self.getLoss(X[i], Y[i])
			lossList.append(loss)
			self.backward(dl)
			self.update()
		return np.mean(lossList)


	def train(self, tX, tY, vX, vY, maxIter, patience):
		maxPatience = patience
		bestLoss = float('inf')
		vLoss = float('inf')
		print('********Start training********')
		totalStart = timeit.default_timer()
		for i in range(maxIter):
			startTime = timeit.default_timer()
			loss = self.epoch(tX, tY)
			endTime = timeit.default_timer()
			print('iter %i, mean loss: %.2f, for %.2fm' %(i, loss, (endTime - startTime)/60.))
			
			if loss < bestLoss:
				bestLoss = loss
				loss = self.epoch(vX, vY)
				print('...validation loss: %.2f' %(loss))
				if loss < vLoss:
					vLoss = loss
					patience = np.min(maxPatience, patience + i/2.)
			else:
				patience = patience - i

			if patience < 0:
				print('no more patience.')
				break
		else:
			print('reach max iter.')

		totalend = timeit.default_timer()
		print('********Finish training********')
		print('best training loss: %.2f, best validation loss: %.2f, for %.2fm' %(bestLoss, vLoss, (totalStart - totalend)/60.))
		return


	def test(self, X, Y):
		loss = self.epoch(X, Y)
		print('test loss: %.2f' %(loss))
		return loss


	def predict(self, X):
		Y = []
		for x in X:
			Y.append(self.forward(x))
		return Y




if __name__ == '__main__':
	des = [('c', (2, 1, 2, (3,3)))]
	ll = buildNet((3,3, 3), des, seedNum = 6983)
	nn = net(ll, 1)
	iData = np.random.rand(3, 3,3)
	oData = nn.forward(iData)
	# print(iData)
	print(oData)
	dl, loss = nn.getLoss(iData, 4)
	print('')
	# print(dl)
	nn.backward(dl)
	nn.update()
	# print(nn.layerList[0].loss)
	print('')
	oData = nn.forward(iData)
	print(oData)