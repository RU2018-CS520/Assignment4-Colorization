import numpy as np

def pad(ix, iy, iz, padSize, iData):
	if padSize == 0:
		return iData
	res = np.zeros((iz, ix+2*padSize, iy+2*padSize), dtype = np.float16)
	res[:, padSize: padSize+ix, padSize: padSize+iy] = iData
	return res


def setNpSeed(seed = 6983):
	np.random.seed(seed)
	return


class lossFunc(object):
	def norm2(o, y):
		dl = o - y
		return (dl, np.square(dl))