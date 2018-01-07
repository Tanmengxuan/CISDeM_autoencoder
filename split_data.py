import pickle
#from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import numpy as np
import autoencoder
import pandas as pd
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

filename  = "data/longRun_0_80Features_fold0mx"

with open(filename, 'rb') as file_handle:

	loaded_data = pickle.load(file_handle, encoding='latin1')
label = loaded_data['label']
data = loaded_data['data']
print (data.shape)



enc, dec = autoencoder.train_auto_encoder(data, load_pre_trained=True)
#print (enc)
x, y, d = autoencoder.encode_data(enc, dec, data)
print (x)
print (x.shape)
print (x[0])

df = pd.DataFrame(x) 
print (df.head(5))
print (df.shape)
#print(list(df))

df2 = pd.DataFrame(label)
#print (df2)
#print (df2.shape)
df2.loc[df2[0] == 0, "attack"] = 1  
df2.loc[df2[0] != 0, "attack"] = -1
#print (df2)

target = df2['attack']
#print (type(target))

model = svm.OneClassSVM(nu=0.02051877661633759, kernel='rbf', gamma=0.00005)  
model.fit(x)
 
preds = model.predict(x)  
targs = target

#preds = list(preds)





print("accuracy: ", metrics.accuracy_score(targs, preds))  
print("precision: ", metrics.precision_score(targs, preds))  
print("recall: ", metrics.recall_score(targs, preds))  
print("f1: ", metrics.f1_score(targs, preds))  
print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))



#preds = list(preds)
#targs = df2["attack"].tolist()
#print (type(targs))

false_positive_rate, true_positive_rate, thresholds =metrics.roc_curve(targs, preds)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

color = ['y', 'b']
inp = [3.0,2.0]
a = np.arange(10)
for i in range(2):
	plt.title('Receiver Operating Characteristic')
	#plt.plot(false_positive_rate, true_positive_rate, 'y',label='AUC = %0.2f'% roc_auc)
	plt.plot(a*inp[i], color[i],label= "hello")
	
	
#	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')



plt.plot(0.9,label ='test')
plt.legend(loc='lower right')
plt.show()



def count_label(label):
	normal_count = 0.0
	abnormal_count = 0.0
	
	
#	label = loaded_data["label"]
#	data = loaded_data["data"]
	
	
	for i in label:
		if i == 0.0:
			normal_count += 1
		else:
			abnormal_count +=1
	
	print ( ("normal :" + str(normal_count))) 
	print ( ("abnormal :" + str(abnormal_count))) 
	print ( ("ratio :" + str(abnormal_count/normal_count))) 
	print  ( ("percentage :" + str(abnormal_count/len(label)))) 

#count_label(label)
#for i in loaded_data:
#	print (len(loaded_data(i)))
#

#print (loaded_data["data"])

#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[3,7],[9,3]])
#y = np.array([0, 0,0, 1, 1,1])
#sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)


#label = loaded_data["label"]
#label = np.array(label)
#data = loaded_data["data"]
#skf = StratifiedKFold(n_splits=4, shuffle=False,  random_state=None)
#
#files = dict()
#counter = 0
#for train_index, test_index in skf.split(data, label):
#	data_test = data[test_index]
#	label_test = label[test_index]
#	label_test = list(label_test)
#	files['label'] = label_test
#	files['data'] = data_test
#	
#	pickle.dump(files,open("longRun_0_80Features_fold"+str(counter)+"mx", "wb"))
#	counter +=1
#	files = dict()
#
	

#print("TRAIN:", train_index, "TEST:", test_index)

#print (X[train_index])
