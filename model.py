import tensorflow as tf
from keras.layers import *
from keras.models import *
import pdb
from keras.utils import plot_model

def CNN(vgg_weight_path):
	input_img = Input(shape=(128, 128, 3))
	#pdb.set_trace()
	x = Permute((1, 3, 2))(input_img)
	x = Permute((2, 1, 3))(x)
	#pdb.set_trace()
	dim_order = "channels_first"
	#begin building VGG network
	x = Conv2D(64, (3, 3), padding="same", activation="relu", name="block1_conv1", data_format=dim_order)(x)
	x = Conv2D(64, (3, 3), padding="same", activation="relu", name="block1_conv2", data_format=dim_order)(x)
	x = MaxPooling2D((2, 2), name="block1_pool", data_format=dim_order)(x)
	o1 = x

	x = Conv2D(128, (3, 3), padding="same", activation="relu", name="block2_conv1", data_format=dim_order)(x)
	x = Conv2D(128, (3, 3), padding="same", activation="relu", name="block2_conv2", data_format=dim_order)(x)
	x = MaxPooling2D((2, 2), name="block_2_pool", data_format=dim_order)(x)
	o2 = x

	x = Conv2D(256, (3, 3), padding="same", activation="relu", name="block3_conv1", data_format=dim_order)(x)
	x = Conv2D(256, (3, 3), padding="same", activation="relu", name="block3_conv2", data_format=dim_order)(x)
	x = Conv2D(256, (3, 3), padding="same", activation="relu", name="block3_conv3", data_format=dim_order)(x)
	x = MaxPooling2D((2, 2), name="block3_pool", data_format=dim_order)(x)
	o3 = x

	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block4_conv1", data_format=dim_order)(x)
	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block4_conv2", data_format=dim_order)(x)
	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block4_conv3", data_format=dim_order)(x)
	x = MaxPooling2D((2, 2), name="block4_pool", data_format=dim_order)(x)
	o4 = x

	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block5_conv1", data_format=dim_order)(x)
	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block5_conv2", data_format=dim_order)(x)
	x = Conv2D(512, (3, 3), padding="same", activation="relu", name="block5_conv3", data_format=dim_order)(x)
	x = MaxPooling2D((2, 2), name="block5_pool", data_format=dim_order)(x)
	o5 = x
	vgg = Model(input_img, x)
	vgg.load_weights(vgg_weight_path)
	x = AveragePooling2D((2, 2), data_format=dim_order)(x)
	#plot_model(Model(input_img, x), show_shapes=True, to_file='mymodel.png')
	return Model(input_img, x)

def Dense_layers():#2 * 2 * 512
	input_img1 = Input(shape=(512, 2, 2), name="input_img1")
	input_img2 = Input(shape=(512, 2, 2), name="input_img2")
	#pdb.set_trace()
	x = Flatten()(input_img1)
	y = Flatten()(input_img2)
	x = Dense(1000, activation="relu")(x)
	y = Dense(1000, activation="relu")(y)
	x = Concatenate()([x, y])
	x = Dense(1000, activation="relu")(x)
	label_pred = Dense(2, activation="softmax")(x)
	bbox_pred = Dense(4, activation="sigmoid")(x)
	outpt = Concatenate()([label_pred, bbox_pred])
	#plot_model(Model([input_img1, input_img2], outpt), show_shapes=True, to_file='mymodel.png')
	return Model([input_img1, input_img2], outpt)
	