import numpy as np

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
		self.loss = self.lossFunc(oData, y)
		return self.loss

	def backward(self, loss):
		if loss is None:
			loss = self.loss

		for layer in reversed(self.layerList):
			loss = layer.backward(loss)
		return loss

	def update(self):
		for layer in self.layerList:
			layer.update(self.learnRate)
		return



if __name__ == '__main__':
	pass