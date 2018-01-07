import tflearn
import tensorflow as tf
import input_data
import keras
import os.path
import numpy as np
from os import mkdir
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, TimeDistributed
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support
framework = "keras"
loadConfig = 1
def loadConfig(configPath=None):
	if not configPath:
		cfg = {
			"input_path" : "../data/fakeMaster_lstm_w*",
			"lstm_size" : 64,
			"n_inputs" : 64,
			"learning_rate" : 0.001,
			"num_features" : 80,
			"drop_rate" : 0.0
		}

	else:
		with open(configPath, "r") as configFile:
			json_str = configFile.read()
			cfg = json.loads(json_str)
	global LSTM_SIZE
	global N_INPUTS
	global INPUT_PATH
	global LEARNING_RATE
	global NUM_FEATURES
	global DROP_RATE

	LSTM_SIZE = cfg["lstm_size"]
	N_INPUTS = cfg["n_inputs"]
	INPUT_PATH = cfg["input_path"]
	LEARNING_RATE = cfg["learning_rate"]
	NUM_FEATURES = cfg["num_features"]
	DROP_RATE = cfg["drop_rate"]
	return cfg
def getNextRunDir(prefix):
	script_path = os.path.dirname(os.path.realpath(__file__))
	output_path = os.path.join(script_path, "../runs/",prefix)
	#Find the directory name in the series which is not used yet
	for num_run in range(0,500000):
		if not os.path.isdir(output_path+'_{}'.format(num_run)):
			mkdir(output_path+'_{}'.format(num_run))
			output_path = output_path+'_{}'.format(num_run)
			break
	return output_path

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

class Metrics(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		val_predict = val_predict.reshape(val_predict.shape[0]*val_predict.shape[1], val_predict.shape[2])
		val_targ = self.validation_data[1]
		val_targ = val_targ.reshape(val_targ.shape[0]*val_targ.shape[1], val_targ.shape[2])
		_val_precision,_val_recall,_val_f1,_support = precision_recall_fscore_support(val_targ,val_predict)

		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print " - val_f1: (%f,%f) - val_precision: (%f,%f) - val_recall (%f,%f) - val_support(%f,%f)" %(_val_f1[0],_val_f1[1], _val_precision[0],_val_precision[1], _val_recall[0],_val_recall[1], _support[0],_support[1])
		return
def printConfig(dir,cfg):
	str = json.dumps(cfg)
	with open(os.path.join(dir,"netConfig.json"),"wb") as f:
		f.write(str)
if framework == "keras":
	configPath = None
	if loadConfig == 1:
		configPath == "curConfig.json"
	cfg = loadConfig(configPath)
	inp = input_data.inputter(INPUT_PATH)
	trainX, trainY = inp.getTrainData(ORLabels=False)
	trainX = trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2])
	print "train data shape", trainX.shape, trainY.shape

	validX, validY = inp.getValidationData(ORLabels=False)
	validX = validX.reshape(validX.shape[0],validX.shape[1],validX.shape[2])
	print "Validation data shape", validX.shape, validY.shape

	model = Sequential()
	model.add(LSTM(LSTM_SIZE, return_sequences=True, input_shape=(N_INPUTS,NUM_FEATURES), dropout = DROP_RATE))
	model.add(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_RATE))
	model.add(TimeDistributed(Dense(2, activation='softmax')))
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
	output_dir = getNextRunDir('lstm_search')
	tbCallBack = keras.callbacks.TensorBoard(log_dir=output_dir, histogram_freq=1, write_grads=True, write_graph= False, write_images=True)
	checkpoint = keras.callbacks.ModelCheckpoint(output_dir+'/best.h5', save_best_only=True)
	metrics = Metrics()
	history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data =(validX, validY), verbose=1,
		callbacks = [early_stop, metrics, tbCallBack, checkpoint])
	printConfig(output_dir, cfg)
	print('Results stored in {}'.format(output_dir))
