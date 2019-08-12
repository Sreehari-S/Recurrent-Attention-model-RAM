import numpy as np
from numpy.random import RandomState
from skimage.transform import resize
import torch

class Resize(object):
	def __init__(self, output_shape, order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=True, anti_aliasing_sigma=None):
		self.kwargs = dict(order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range, anti_aliasing=anti_aliasing, anti_aliasing_sigma=anti_aliasing_sigma)
		self.output_shape = output_shape	  
	def __call__(self, image):
		return resize(image, (self.output_shape,)*(len(image.shape) -1) if type(self.output_shape) is int else self.output_shape, **self.kwargs)

class ToTensor(object):
	def __init__(self):
		pass
	def __call__(self, array):
		return torch.tensor(np.moveaxis(array, -1, 0))

class RandomHorizontalFlip(object):
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self, array):
		if np.random.rand()<=self.p:
			return np.fliplr(array)
		else:
			return array

class RandomVerticalFlip(object):
	def __init__(self, p=0.5, seed=1):
		self.p = p
	def __call__(self, array):
		if np.random.rand()<=self.p:
			return np.flipud(array)
		else:
			return array
		