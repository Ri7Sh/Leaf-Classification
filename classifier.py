import warnings
warnings.filterwarnings("ignore")
import os
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn import svm
from neupy import algorithms, environment
from neupy import algorithms, layers, environment
from neupy.exceptions import StopTraining
import theano

def preprocess():
	'''
		this function preprocess the input, normalize and split it into 
		training and test data
	'''
	#print 'loading the dataset'
	flavia_data = pd.read_csv("Flavia_features.csv")
	images_files = os.listdir("./leaf")
	#print 'creating labels of images'
	split_points = [1001,1059,1060,1122,1552,1616,1123,1194,1195,1267,1268,1323,1324,1385,1386,1437,1497,1551,1438,1496,2001,2050,2051,2113,2114,2165,2166,2230,2231,2290,2291,2346,2347,2423,2424,2485,2486,2546,2547,2612,2616,2675,3001,3055,3056,3110,3111,3175,3176,3229,3230,3281,3282,3334,3335,3389,3390,3446,3447,3510,3511,3563,3566,3621]
	target_labels = []
	for img_file in images_files:
		target_value = int(img_file.split(".")[0])
		i=0
		flag=0
		for i in range(0,len(split_points),2):
		  #  print i
			if((target_value>=split_points[i]) and (target_value<=split_points[i+1])):
				flag=1
				break
		if flag == 1:
			t = int((i/2))
			target_labels.append(t)
	   # print '*******************'
	arr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for x in target_labels:
		arr[x]=arr[x]+1
	#print arr
	y = np.array(target_labels)
	X = flavia_data.iloc[:,1:]
	#print X
	#print y
	#print 'train test split'
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 41)

	#print X_train.shape
	#print type(X_train)
	#print X_test.shape
	#print y_train.shape
	#print y_test.shape


	#print X_train
	#print y_train
	
	#print 'doing feature normalization'
	scaler =  StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	#print type(X_train)
	#print type(X_test)
	#print type(y_train)
	return X_train,X_test,y_train,y_test


def PNN(X_train, X_test, y_train, y_test,X_dummy):
	environment.reproducible()
	pnn = algorithms.PNN(std=0.1, verbose=False)
	pnn.train(X_train, y_train)
	#print 'done trainin'
	y_predicted = pnn.predict(X_test)
	y_dummy = pnn.predict(X_dummy)
	#print y_predicted
	return y_dummy,y_predicted,metrics.accuracy_score(y_test, y_predicted)

def SVM(X_train, X_test, y_train, y_test,X_dummy):
	clf = svm.SVC()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	y_dummy = clf.predict(X_dummy)
	#print y_pred
	return y_dummy,y_pred,metrics.accuracy_score(y_test, y_pred)

def ANN(X_train, X_test, y_train, y_test,X_dummy):
	environment.reproducible()
	target_scaler = OneHotEncoder()
	net = algorithms.Momentum(
		[
			layers.Input(17),
			layers.Relu(100),
			layers.Relu(70),
			layers.Softmax(32),
		],
		error='categorical_crossentropy',
		step=0.01,verbose=True,shuffle_data=True,
		momentum=0.99,nesterov=True,
	)
	# converting vector to one hot encoding
	d1 =int(y_train.shape[0])
	d2=int(y_test.shape[0])
	Y_train = np.zeros((d1,32))
	Y_test = np.zeros((d2,32))
	Y_train[np.arange(d1),y_train]=1
	Y_test[np.arange(d2),y_test]=1

	net.architecture()
	net.train(X_train, Y_train, X_test, Y_test, epochs=20)
	y_predicted = net.predict(X_test).argmax(axis=1)
	y_dummy = net.predict(X_dummy).argmax(axis=1)
	#print 'predicted values'
	#print y_predicted
	Y_test = np.asarray(Y_test.argmax(axis=1)).reshape(len(Y_test))
	#print(metrics.classification_report(Y_test, y_predicted))
	return y_dummy,y_predicted,metrics.accuracy_score(Y_test, y_predicted)

def dummy_data():
	data = pd.read_csv("dummy.csv")
	#print data
	X = data.iloc[:,0:]
	#print '********************'
	#print X.shape
	scaler =  StandardScaler()
	X = scaler.fit_transform(X)
	return X
	

X_train, X_test, y_train, y_test = preprocess()
X_dummy = dummy_data(); 
s = raw_input("enter the name of the classifier: PNN or ANN or SVM")
if s=='PNN' :
	y_dum,y,score= PNN(X_train, X_test, y_train, y_test,X_dummy)
elif s=='ANN':
	y_dum,y,score= ANN(X_train, X_test, y_train, y_test,X_dummy)
else :
	y_dum,y,score= SVM(X_train, X_test, y_train, y_test,X_dummy)
print 'accuracy of classifier is ==>> '
print score
print '***************************************************************'
print 'dummy predicted values'
print '***************************************************************'
print y_dum

