import cv2
import numpy as np
import copy
import random
import pdb

def get_rand_color():
	return random.randint(20, 235), random.randint(20, 235), random.randint(20, 235)

def visualizing(img, bounding_box, id_to_name, ori_shape):
	"""
	draw bounding boxes and their corresponding names on the given image
	img: numpy array of the image to be shown
	bounding_box: list/numpy array. each element is a tuple(id, x_min, y_min, width, height, prob) describing one box to be shown
	"""
	image = copy.deepcopy(img)
	#pdb.set_trace()
	cur_width = img.shape[0]
	cur_height = img.shape[1]
	ori_width, ori_height = ori_shape
	#pdb.set_trace()
	image = cv2.resize(image, (ori_height, ori_width))
	for box in bounding_box:
		idx, x_min, y_min, x_max, y_max, prob = box
		x_min, y_min = y_min, x_min
		x_max, y_max = y_max, x_max
		width = x_max - x_min
		height = y_max - y_min
		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), get_rand_color(), 2)
		#text = id_to_name[idx] + ":" + "%.3f" % prob
		#ret, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		#cv2.rectangle(image, (x_min, y_min), (x_min + ret[0], y_min + ret[1] + baseline), (255, 255, 255), -1)
		#cv2.putText(image, text, (x_min, y_min + ret[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	
	cv2.imshow("image_with_bounding_box", image)
	cv2.waitKey(0)
