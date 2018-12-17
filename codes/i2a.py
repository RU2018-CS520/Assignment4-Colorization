import numpy as np
from PIL import Image

def i2a(path, index):
	img = Image.open(path + 'img' + index + '.jpg')
	npa = np.array(img)
	npa = 1.0 * npa / 255.0
	return np.transpose(npa, (2,0,1))

def batch(path, s, e):
	length = e - s + 1
	res = np.empty((length, 3, 32, 32), dtype = np.float64)
	y = np.array([], dtype = np.int64)
	for i in range(s, e+1):
		res[i-1] = i2a(path, str(i))
		y = np.append(y, 0)
	return (res, y)

if __name__ == '__main__':
	res, y = batch('data/', 1, 20)
	print(res[0, :, :])