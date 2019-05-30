from keras.models import *
from keras.layers import *
from keras.optimizers import *
from model import *
from keras import backend as K
#from data_gen import *
import numpy as np
import copy
import sys
from keras.applications.vgg16 import preprocess_input
import time
#from image_visualizing import *
import cv2

class predictor(object):
	def readin(self):
		fin = open("img.temp", "r")
		width, height = map(int, fin.readline().split())
		image = np.zeros((width, height, 3))
		for i in range(width):
			for j in range(height):
				image[i][j][0], image[i][j][1], image[i][j][2] = map(int, fin.readline().split())
		fin.close()
		return image

	def data_parser(self, data):
		l = len(data)
		width = data[l - 2]
		height = data[l - 1]
		image = np.array(data[0: l - 2], dtype=np.float64)
		image = np.reshape(image, (width, height, 3))
		#image.astype(np.float64)
		return image

	def enhance(self, image):
		threshold = 0
		height = image.shape[1]
		if image.mean() > threshold:
			return image[:, height // 8: height - height // 8, :]
		for i in range(5):
			image = np.clip(image * 1.1, 0, 255)
			if image.mean() > threshold:
				return image[:, height // 8: height - height // 8, :]

	def __init__(self):
		self.path = "path.temp"
		self.length = 128
		self.threshold = 0.5
		vgg_weight_path = "data/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
		"""
		above are hyperparameters
		"""
		self.vgg_feature_extractor = CNN(vgg_weight_path)
		fin = open(self.path, "r")
		path = fin.readline().strip("\n")
		fin.close()
		template = cv2.imread(path, 1)
		#template = self.readin()
		template = np.array([cv2.resize(template, (self.length, self.length))])
		template = preprocess_input(template) / 50
		self.outpt_template = self.vgg_feature_extractor.predict_on_batch(template)
		self.predictTF = Dense_layers()
		self.predictTF.load_weights("mymodel3.h5")
		

	def query_TF(self):
		self.path = "path.temp"
		fin = open(self.path, "r")
		path = fin.readline().strip("\n")
		fin.close()
		#print(1)
		image = cv2.imread(path, 1)
		image = self.enhance(image)
		#cv2.imshow("1", image)
		#cv2.waitKey(0)
		#image = self.data_parser(data)
		image = np.array([cv2.resize(image, (self.length, self.length))])
		image = preprocess_input(image) / 50
		outpt_image = self.vgg_feature_extractor.predict_on_batch(image)
		inpt = {"input_img1": self.outpt_template, "input_img2": outpt_image}
		score = self.predictTF.predict_on_batch(inpt)[0][1]
		print(score)
		fout = open("rslt.temp", "w")
		fout.write("%f\n" % score)#fout.write("%d\n" % (1 if score > self.threshold else 0))
		fout.close()

	def query_pos(self):
		self.path = "path.temp"
		fin = open(self.path, "r")
		path = fin.readline().strip("\n")
		fin.close()
		image = cv2.imread(path, 1)
		#image = self.data_parser(data)
		shape = image.shape[0], image.shape[1]
		outpt_img = []
		for i in range(9):
			x = i // 3
			y = i % 3
			im = cv2.resize(image[shape[0] * (5 * x + 2) // 20: shape[0] * (5 * x + 8) // 20, shape[1] * (5 * y + 2) // 20:shape[1] * (5 * y + 8) // 20, :], (self.length, self.length))
			im = np.array([preprocess_input(im) / 50])
			outpt_img.append(self.vgg_feature_extractor.predict_on_batch(im))
		score = [0] * 9
		for i in range(9):
			score[i] = 1
			inpt = {"input_img1": self.outpt_template, "input_img2": outpt_img[i]}
			score[i] = min(score[i], self.predictTF.predict_on_batch(inpt)[0][1])
		mxp = np.argmax(np.array(score))
		fout = open("rslt.temp", "w")
		fout.write("%d\n" % mxp)
		fout.close()
###################################################################

def predict(vgg_feature_extractor, predictor, template_file, image_file):
	template = cv2.imread(template_file, 1)
	template = np.array([cv2.resize(template, (length, length))])
	#cv2.imshow("a", template[0])
	#cv2.waitKey(0)
	template = preprocess_input(template) / 50
	#outpt_template = vgg_feature_extractor.predict_on_batch(template)
	image = cv2.imread(image_file, 1)
	shape = image.shape[0], image.shape[1]
	outpt_img = []
	for i in range(9):
		x = i // 3
		y = i % 3
		im = cv2.resize(image[shape[0] * (5 * x + 2) // 20: shape[0] * (5 * x + 8) // 20, shape[1] * (5 * y + 2) // 20:shape[1] * (5 * y + 8) // 20, :], (length, length))
		#cv2.imshow("1", im)
		#cv2.waitKey(0)
		im = np.array([preprocess_input(im) / 50])
		outpt_img.append(vgg_feature_extractor.predict_on_batch(im))
	"""
	score = [0] * 9
	for i in range(9):
		score[i] = 1
		inpt = {"input_img1": outpt_template, "input_img2": outpt_img[i]}
		score[i] = min(score[i], predictor.predict_on_batch(inpt)[0][1])
		#inpt = {"input_img1": outpt_img[i], "input_img2": outpt_template}
		#score[i] = min(score[i], predictor.predict_on_batch(inpt)[0][1])
	mxp = np.argmax(np.array(score))
	x = mxp // 3
	y = mxp % 3
	if score[mxp] < 0.5:
		print("No")
		#visualizing(image, [[0, 1, 7, 1, 7, 1]], 0, shape)
	else:
		print("Yes")
		#visualizing(image, [[0, shape[0] * (5 * x + 2) // 20,  shape[1] * (5 * y + 2) // 20, shape[0] * (5 * x + 8) // 20, shape[1] * (5 * y + 8) // 20, 1]], 0, shape)
	#image = image[x1:x2, y1:y2, :]
	#image = np.array([cv2.resize(image, (length, length))])
	#cv2.imshow("a", image[0])
	#cv2.waitKey(0)
	return score[mxp] >= 0.5, x, y
	"""
	
"""
vgg_feature_extractor = CNN(vgg_weight_path)
predictor = Dense_layers()
predictor.load_weights("mymodel3.h5")
template_file = "template5.png"
image_file = "picture11.jpeg"
st_time = time.time()
ret_val = predict(vgg_feature_extractor, predictor, template_file, image_file)
print((time.time() - st_time))
print(ret_val)
"""
