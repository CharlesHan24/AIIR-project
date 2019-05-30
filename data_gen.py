import cv2
import numpy as np
import zipfile
import random
import glob
import copy
import pdb
from image_visualizing import *
from keras.applications.vgg16 import preprocess_input

vgg_weight_path = "data/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
img_file = "data/val2017.zip"
label_file = "data/bounding_box.dat"
tra_val = 89
batch_size = 16
learning_rate = 0.005
category_num = 90
length = 128
load_weight_bool = False
epoch = 30
id_to_name = []
for i in range(category_num):
	id_to_name.append(str(i))
warning_sign = glob.glob("warning_sign/" + "*") #+ glob.glob("key/" + "*.jpeg") + glob.glob("key/" + "*.png")
warehouse = glob.glob("warehouse/" + "*")# + glob.glob("floor/" + "*.jpeg") + glob.glob("floor" + "*.png")

def collect(label_file, category_num):
	ret_bin = [[] for i in range(category_num)]
	fin = open(label_file, "r")
	for i in range(36781):
		idx = [fin.readline().strip("\n")]
		idx += list(map(float, fin.readline().strip("\n").split(" ")))
		for j in range(1, 5):
			idx[j] = int(idx[j])
		idx[1], idx[2] = idx[2], idx[1]
		idx[3], idx[4] = idx[4], idx[3]
		label = int(fin.readline().strip("\n"))
		ret_bin[label - 1].append(copy.deepcopy(idx))
		_ = fin.readline()
	return ret_bin

def find_image(object_bin, zipfin, category, length, min_length):
	if category == -1:
		category = random.randint(0, 89)
		while len(object_bin[category]) == 0:
			category = random.randint(0, 89)
		idx = object_bin[category][random.randint(0, len(object_bin[category]) - 1)][0]
			#if int(idx) >= l and int(idx) <= r:
			#	break
		data = zipfin.read("val2017" + "/" + idx.zfill(12) + ".jpg")
		img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
		cropped_length = random.randint(min(length, min(img.shape[0], img.shape[1]) // 2), min(img.shape[0], img.shape[1]))
		stx = random.randint(0, img.shape[0] - cropped_length)
		sty = random.randint(0, img.shape[1] - cropped_length)
		ret_img = cv2.resize(img[stx:stx + cropped_length, sty:sty + cropped_length, :], (length, length))
		return ret_img, 0, 0, 0, 0
	else:
		idx = 0
		while True:
			idx = random.randint(0, len(object_bin[category]) - 1)
			name = object_bin[category][idx][0]
			#if int(name) < l or int(name) > r:
			#	continue
			if object_bin[category][idx][3] >= min_length and object_bin[category][idx][4] >= min_length:
				break
		if min_length != length or random.random() < 0.7:
			data = zipfin.read("val2017" + "/" + object_bin[category][idx][0].zfill(12) + ".jpg")
			img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
		else:
			img = cv2.imread("images/%d.jpg" % (category + 1), 1)
			return cv2.resize(img, (length, length)), 0, 0, 1, 1
		#cv2.imshow("image", img)
		#cv2.waitKey(0)

		mx = max(object_bin[category][idx][3], object_bin[category][idx][4])
		mn = min(object_bin[category][idx][3], object_bin[category][idx][4])
		x1 = object_bin[category][idx][1]
		y1 = object_bin[category][idx][2]
		x2 = object_bin[category][idx][3] + x1
		y2 = object_bin[category][idx][4] + y1
		if min_length != length:
			lenx = min(img.shape[0], random.randint(max(36, mx), max(2 * mn, mx)))
			leny = min(img.shape[1], max(lenx, mx))
		else:
			lenx = min(img.shape[0], mx)
			leny = min(img.shape[1], mx)
		stx = max(0, min(img.shape[0] - lenx, x1 - (lenx - object_bin[category][idx][3]) // 2))
		sty = max(0, min(img.shape[1] - leny, y1 - (leny - object_bin[category][idx][4]) // 2))
		ret_img = cv2.resize(img[stx:stx + lenx, sty:sty + leny, :], (length, length))
		return ret_img, (x1 - stx) / lenx, (y1 - sty) / leny, (x2 - x1) / (lenx + stx - x1), (y2 - y1) / (leny + sty - y1)

def keys_image(object_bin, zipfin, category_num, length, min_length, batch_size):
	lkeys = len(keys)
	lfloor = len(floor)
	ret_template = cv2.imread(keys[random.randint(0, lkeys - 1)], 1)
	ret_template = cv2.resize(ret_template, (length, length))
	ret_template = preprocess_input(ret_template) / 50
	ret_image = []
	ret_label = []

	for i in range(batch_size):
		ra = random.random()
		if ra < 0.5:
			ret_image.append(cv2.resize(cv2.imread(keys[random.randint(0, lkeys - 1)], 1), (length, length)))
			#cv2.imshow("1", ret_image[i])
			#cv2.waitKey(0)
			ret_label.append([0, 1, 0, 0, 0, 0])
		elif ra < 0.75:
			ret_image.append(cv2.resize(cv2.imread(floor[random.randint(0, lfloor - 1)], 1), (length, length)))
			#cv2.imshow("1", ret_image[i])
			#cv2.waitKey(0)
			ret_label.append([1, 0, 0, 0, 0, 0])
		else:
			ret_image.append(find_image(object_bin, zipfin, -1, length, min_length)[0])
			#cv2.imshow("1", ret_image[i])
			#cv2.waitKey(0)
			ret_label.append([1, 0, 0, 0, 0, 0])
		ret_image[i] = preprocess_input(ret_image[i]) / 50
	return np.array(ret_image), np.array([ret_template]), np.array(ret_label, dtype=np.float32)

def specific_data(object_bin, zipfin, category_num, length, batch_size):
	template = cv2.imread("images/91.png", 1)
	template = cv2.resize(template, (length, length))
	template = np.array([preprocess_input(template) / 50])
	ret_image = []
	label = []
	for i in range(batch_size):
		#background
		if random.random() < 1:
			random.shuffle(warehouse)
			backgr = cv2.imread(warehouse[0], 1)
		else:
			backgr = find_image(object_bin, zipfin, -1, length, length)[0]
		if backgr.shape[0] < backgr.shape[1]:
			backgr = backgr[0:backgr.shape[0], max(0, (backgr.shape[1] - backgr.shape[0]) // 2):min(backgr.shape[1], (backgr.shape[1] + backgr.shape[0]) // 2), :]
		else:
			backgr = backgr[max(0, (backgr.shape[0] - backgr.shape[1]) // 2):min(backgr.shape[0], (backgr.shape[1] + backgr.shape[0]) // 2), 0:backgr.shape[1], :]	
		backgr = cv2.resize(backgr, (backgr.shape[0], backgr.shape[0]))

		if random.random() < 0.5:
			backgr = cv2.resize(backgr, (length, length))
			if random.random() < 0.5:
				backgr = cv2.flip(backgr, 1)
			#cv2.imshow("1", backgr)
			#cv2.waitKey(0)
			ret_image.append(preprocess_input(backgr) / 50)
			label.append([1, 0, 0, 0, 0, 0])
		else:
			#add positive sample to the background(a kind of replacement)
			random.shuffle(warning_sign)
			sample = cv2.imread(warning_sign[0], 1)
			mn = min(min(sample.shape[0], sample.shape[1]), backgr.shape[0])
			if mn > backgr.shape[0] // 6:
				mn = random.randint(backgr.shape[0] // 6, min(mn, backgr.shape[0] * 4 // 6))
			sample = cv2.resize(sample, (mn, mn))
			if random.random() < 0.5:
				sample = cv2.flip(sample, random.randint(0, 1))
			if random.random() < 0.5:
				backgr = cv2.flip(backgr, 1)
			stx = random.randint(0, backgr.shape[0] - mn)
			sty = random.randint(0, backgr.shape[0] - mn)
			backgr[stx:stx + mn, sty:sty + mn, :] = copy.deepcopy(sample)
			label.append([0, 1, stx / backgr.shape[0], sty / backgr.shape[1], mn / (backgr.shape[0] - stx), mn / (backgr.shape[1] - sty)])
			#visualizing(backgr, [[category_num, int(backgr.shape[0] * label[i][2]), int(backgr.shape[1] * label[i][3]), int(backgr.shape[0] * label[i][2] + label[i][4] * backgr.shape[0] * (1 - label[i][2])), int(backgr.shape[1] * label[i][3] + label[i][5] * backgr.shape[1] * (1 - label[i][3])), 1]], id_to_name, (backgr.shape[0], backgr.shape[1]))
			backgr = cv2.resize(backgr, (length, length))
			#cv2.imshow("1", backgr)
			#cv2.waitKey(0)
			ret_image.append(preprocess_input(backgr) / 50)

	return template, np.array(ret_image), np.array(label) 

def batch_loader(img_file, label_file, l, r, length=128, batch_size=16, category_num=90):
	object_bin = collect(label_file, category_num)
	zipfin = zipfile.ZipFile(img_file)

	ret_template = None
	ret_image = []
	ret_label = []
	category = random.randint(l - 1, r - 1)
	cnt = 0
	while True:
		if cnt == 0:
			if random.random() < 1:
				_1, _2, _3 = specific_data(object_bin, zipfin, category_num, length, batch_size)
				yield _2, _1, _3
				continue

			category = random.randint(l - 1, r - 1)
			while len(object_bin[category]) == 0:
				category = random.randint(l - 1, r - 1)
			ret_template, _1, _2, _3, _4 = find_image(object_bin, zipfin, category, length, length)
			ret_template = preprocess_input(ret_template) / 50
			#cv2.imshow("imag", ret_template)
			#cv2.waitKey(0)
		cnt += 1
		cur_label = random.randint(0, 1)
		image, x1, y1, x2, y2 = find_image(object_bin, zipfin, category if cur_label == 1 else -1, length, length // 5,)
		image = preprocess_input(image) / 50
		#visualizing(image, [[category, int(image.shape[0] * x1), int(image.shape[1] * y1), int(image.shape[0] * x1 + x2 * image.shape[0] * (1 - x1)), int(image.shape[1] * y1 + y2 * image.shape[1] * (1 - y1)), 1]], id_to_name, (image.shape[0], image.shape[1]))
		ret_image.append(image)
		label = np.zeros(6)
		label[cur_label] = 1
		if cur_label == 1:
			label[2], label[3], label[4], label[5] = x1, y1, x2, y2
		ret_label.append(label)
		if cnt == batch_size:
			yield np.array(ret_image), np.array([ret_template]), np.array(ret_label)
			cnt = 0
			ret_image = []
			ret_label = []

if __name__ == "__main__":
	loader = batch_loader(img_file, label_file, tra_val, 90, length, batch_size, category_num)
	x, y, z = next(loader)
