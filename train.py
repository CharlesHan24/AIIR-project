from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import *
from model import *
from keras import backend as K
from data_gen import *
import numpy as np
import copy
import sys
import random
import time

def loss_func(y_true, y_pred, lmb=0):
	l1 = K.categorical_crossentropy(y_true[:, 0:2], y_pred[:, 0:2])
	l2 = lmb * K.sum(K.square(y_true[:, 2:6] - y_pred[:, 2:6]), axis=1)
	return l2 + l1

def loss_np(y_true, y_pred, lmb=0):
	y_pred[:, 0:2] = np.clip(y_pred[:, 0:2], 1e-9, 0.999999999)
	l1 = -np.mean(np.sum(y_true[:, 0:2] * np.log(y_pred[:, 0:2]), axis=1))
	l2 = lmb * np.mean(np.sum((y_true[:, 2:6] - y_pred[:, 2:6]) ** 2, axis=1))
	return l2 + l1

def my_metric(y_true, y_pred):
	return K.mean(K.sum(K.round(y_pred[:, 0:2] * y_true[:, 0:2]), axis=1))

def my_metric_np(y_true, y_pred):
	return np.mean(np.sum(np.round(y_pred[:, 0:2] * y_true[:, 0:2]), axis=1))

def my_update(y_true, y_pred, tp, tn, fp, fn, batch_size):
	for i in range(batch_size):
		if y_true[i][1] > 0.5:
			if y_pred[i][1] > threshold:
				tp += 1
			else:
				tn += 1
		else:
			if y_pred[i][1] > threshold:
				fp += 1
			else:
				fn += 1
	return tp, tn, fp, fn


def training(data_gen, step, mode):
	loss_sum = 0
	acc_sum = 0
	tp, tn, fp, fn = 0, 0, 0, 0
	for j in range(step):
		time.sleep(0.1)
		img, template, label = next(data_gen)
		outpt_img = vgg_feature_extractor.predict_on_batch(img)
		outpt_template = vgg_feature_extractor.predict_on_batch(template)
		for t in range(batch_size - 1):
			outpt_template = np.append(outpt_template, np.array([copy.deepcopy(outpt_template[0])]), axis=0)
		for t in range(batch_size - 1):
			if random.random() < 0.5:
				outpt_img[t], outpt_template[t] = copy.deepcopy(outpt_template[t]), copy.deepcopy(outpt_img[t])
		inpt = {"input_img1": outpt_img, "input_img2": outpt_template}
		
		if mode == "training":
			ret_val = predictor.train_on_batch(inpt, label)
			loss_sum += ret_val[0].mean()
			acc_sum += ret_val[2].mean()
		else:
			ret_val = predictor.predict_on_batch(inpt)
			loss_sum += loss_np(label, ret_val)
			acc_sum += my_metric_np(label, ret_val)
			tp, tn, fp, fn = my_update(label, ret_val, tp, tn, fp, fn, batch_size)

		sys.stdout.write('\r')
		sys.stdout.write("loss=%f, acc=%f " % (loss_sum / (j + 1), acc_sum / (j + 1) * 100))
		sys.stdout.write("%s%% |%s" % (int(j * 100 // step), (j * 100 // step + 1) * "#"))
		sys.stdout.flush()
	if mode == "validation":
		print("true_positive rate = %f, false_negative rate = %f" % (tp / (tp + tn), fn / (fp + fn)))
"""
hyperparameters
"""
vgg_weight_path = "data/vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
img_file = "data/val2017.zip"
label_file = "data/bounding_box.dat"
tra_val = 89
batch_size = 1
learning_rate = 0.0005
category_num = 90
length = 128
load_weight_bool = True
epoch = 30
training_step = 1000
validation_step = 200
threshold = 0.5

vgg_feature_extractor = CNN(vgg_weight_path)
predictor = Dense_layers()
for layer in predictor.layers:
	layer.trainable = False
predictor.layers[-2].trainable = True
if load_weight_bool == True:
	predictor.load_weights("mymodel3.h5")
#print(predictor.summary())
training_data = batch_loader(img_file, label_file, 1, tra_val - 1, length, batch_size, category_num)
validation_data = batch_loader(img_file, label_file, tra_val, 90, length, batch_size, category_num)

predictor.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=loss_func, metrics=["accuracy", my_metric])

for i in range(epoch):
	print("The %dth epoch begins" % i)
	#training(training_data, training_step, mode="training")
	predictor.save_weights("mymodel3.h5")
	print("Training has completed, now validation starts")
	training(validation_data, validation_step, mode="validation")
	print("Validation has completed")
