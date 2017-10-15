import pickle
#from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import numpy as np
import autoencoder
import pandas as pd
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC

block1 = [1, 10057,10057]
block2 = [10058, 10166,109]
block3 = [10167, 10255,89]
block4 = [10256, 10347,92]
block5 = [10348, 10358,11]
block6 = [10359, 10497,139]
block7 = [10498, 10717,220]
block8 = [10718, 10799,82]
block9 = [10800, 20660,9861]

def splitframe_data (original_dataframe, block):
    end = block[1] - block[2]%30 
    finallength = block[2] - block[2]%30 
#     print(end)
#     print (finallength)
#     print (end - (block[0]-1))
    df = original_dataframe[block[0]-1:end]
#     print (df)
    listofframes = []
    
    for i in range(0,finallength,30):
        temp = df[i:i+30]
        temp1 = pd.DataFrame(temp.values.reshape(1, -1))
        listofframes.append(temp1)
    result = pd.concat(listofframes)
    
    return result

def splitframe_label (original_dataframe, block):
    
    finallength = block[2] - block[2]%30
    finallength = (finallength)/30
    finallength = int(finallength)
#     print (finallength)
    df = original_dataframe[block[0]-1:(block[0]-1 + finallength)]
    return df

#def convert_label(label):
#	for i in range (len(label)):
#		if label[i] == 1:
#			label[i] == 0
#		else:
#			label[i] == 1
#	return label

def one_class_svm_noab(data_train, data_test,label_train, label_test ):

	df_data_train = pd.DataFrame(data_train)
	df_label_train = pd.DataFrame(label_train)
	df_label_train.rename(columns = {0:'label'}, inplace = True)

	df_data_train = df_data_train.join(df_label_train)
    
	df_data_test = pd.DataFrame(data_test)
	df_label_test = pd.DataFrame(label_test)
	df_label_test.rename(columns = {0:'label'}, inplace = True)
	    
	df_data_test = df_data_test.join(df_label_test)

	cond = df_data_train.label== 1
	rows = df_data_train.loc[cond, :]
	df_data_test = df_data_test.append(rows, ignore_index=True)
	df_data_train.drop(rows.index, inplace=True)

	df_data_train = df_data_train.drop('label', axis = 1)
	df_label_test = df_data_test['label']
	df_label_test = pd.DataFrame(df_label_test)
	df_data_test = df_data_test.drop('label', axis = 1)

	print ("data train shape is : [{}]".format(df_data_train.shape))
	print ("data test shape is : [{}]".format(df_data_test.shape))
	print ("data test label shape is : [{}]".format(df_label_test.shape))
	
	df_label_test.loc[df_label_test['label'] == 0, "attack"] = 1  
	df_label_test.loc[df_label_test['label'] != 0, "attack"] = -1
	target_test = df_label_test['attack']
	                                                              
	model= svm.OneClassSVM(nu=0.2, kernel='rbf', gamma=0.00001)
	model.fit(df_data_train)
	
	#pred_test = model.predict(df_data_test)
	pred_test = model.decision_function(df_data_test)	
	return target_test, pred_test

def one_class_svm(data_train, data_test,label_train, label_test ):

	df_data_train = pd.DataFrame(data_train)
	df_label_train = pd.DataFrame(label_train)
	df_label_train.rename(columns = {0:'label'}, inplace = True)

	   
	df_data_test = pd.DataFrame(data_test)
	df_label_test = pd.DataFrame(label_test)
	df_label_test.rename(columns = {0:'label'}, inplace = True)
	    
	print ("data train shape is : [{}]".format(df_data_train.shape))
	print ("data test shape is : [{}]".format(df_data_test.shape))
	print ("data test label shape is : [{}]".format(df_label_test.shape))
	
	df_label_test.loc[df_label_test['label'] == 0, "attack"] = 1  
	df_label_test.loc[df_label_test['label'] != 0, "attack"] = -1
	target_test = df_label_test['attack']
	                                                              
	model= svm.OneClassSVM(nu=0.2, kernel='rbf', gamma=0.00001)
	model.fit(df_data_train)
	
	#pred_test = model.predict(df_data_test)
	pred_test = model.decision_function(df_data_test)	
	return target_test, pred_test


def svm_svc(data_train, data_test,label_train, label_test):

	clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

	clf.fit(data_train, label_train)

	pred_test = clf.predict_proba(data_test)
	pred_test= np.delete(pred_test, 0, 1)
#	pred_test = pred_test.ravel()
	
	label_test = label_test
	return label_test, pred_test

def create_temporal(data, label):
	pd_data = pd.DataFrame(data)
	pd_label = pd.DataFrame(label)
	
	df1 = splitframe_data(pd_data, block1)
	df2 = splitframe_data(pd_data, block2)
	df3 = splitframe_data(pd_data, block3)
	df4 = splitframe_data(pd_data, block4)
	# df5 = splitframe_data(pd_data, block5)
	df6 = splitframe_data(pd_data, block6)
	df7 = splitframe_data(pd_data, block7)
	df8 = splitframe_data(pd_data, block8)
	df9 = splitframe_data(pd_data, block9)
	
	final_data = pd.concat([df1,df2,df3,df4,df6,df7,df8,df9])
	data =np.array( final_data)
	
	df1_l = splitframe_label(pd_label, block1)
	df2_l= splitframe_label(pd_label, block2)
	df3_l= splitframe_label(pd_label, block3)
	df4_l= splitframe_label(pd_label, block4)
	# df5 = splitframe_data(pd_data, block5)
	df6_l= splitframe_label(pd_label, block6)
	df7_l= splitframe_label(pd_label, block7)
	df8_l= splitframe_label(pd_label, block8)
	df9_l= splitframe_label(pd_label, block9)
	
	final_label = pd.concat([df1_l,df2_l,df3_l,df4_l,df6_l,df7_l,df8_l,df9_l])
	label = np.array(final_label)
	label = label.ravel()

	return data, label

########################################### Reading the data ##################################

filename  = "data/longRun_0_80Features"

with open(filename, 'rb') as file_handle:
	loaded_data = pickle.load(file_handle, encoding='latin1')

label = loaded_data["label"]
label = np.array(label)
data = loaded_data["data"]
#data = np.delete(data,np.s_[20:80], axis=1) #for 2 sub-trees

#data, label = create_temporal(data, label)

enc, dec = autoencoder.train_auto_encoder(data, load_pre_trained=True)
enc_data, y, d = autoencoder.encode_data(enc, dec, data)

print ("original data shape is : [{}]".format(data.shape))
print ("enc data shape is : [{}]".format(enc_data.shape))
mean_f1 = 0
counter = 0
mean_ROC = 0
color = ['darkorange', 'indigo', 'seagreen', 'blue','cyan']

skf = StratifiedKFold(n_splits=5, shuffle=True,  random_state=0)
for train_index, test_index in skf.split(enc_data, label):
	data_train , data_test = data[train_index], data[test_index]
	label_train , label_test = label[train_index], label[test_index]


	target_test, pred_test = one_class_svm(data_train, data_test,label_train, label_test)
	#target_test, pred_test = one_class_svm_noab(data_train, data_test,label_train, label_test)
	#target_test, pred_test = svm_svc(data_train, data_test, label_train, label_test)

	print("area under curve (auc): ", metrics.roc_auc_score(target_test, pred_test))
	print("\n")
#	print("f1: ", metrics.f1_score(target_test, pred_test))  
	#print("area under curve (auc): ", metrics.roc_auc_score(label_test, pred_test))
	
#	f1 =metrics.f1_score(target_test, pred_test)
#	mean_f1 +=f1
	false_positive_rate, true_positive_rate, thresholds =metrics.roc_curve(target_test, pred_test)
	roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

	plt.title('ROC on cross-validation for 8 sub trees (one_class_svm)')
	plt.plot(false_positive_rate, true_positive_rate, color[counter],label='AUC = %0.2f'% roc_auc)
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')

	counter +=1
	mean_ROC += roc_auc
	
mean_ROC = mean_ROC/5
print ("Mean ROC is: " +str(mean_ROC))
print("\n")
#mean_f1 = mean_f1/5
plt.plot(0,"w",label ='mean ROC = %0.2f'% mean_ROC)
#plt.plot(0,"w",label ='mean f1 = %0.2f'% mean_f1)
plt.legend(loc='lower right')
plt.show()









