# sudo pip install matplotlib
# sudo apt-get install python-tk
import numpy as np
import string
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

# *******************************************************************************
# *										*
# *				Question One	 				*
# *										*
# *******************************************************************************

file = open('one.txt', 'r')
oneData = file.read()
oneData = oneData.split('\n')
oneData.pop()
temp = []
for i in oneData:
	temp.append(i.split(','))

oneDesc = np.array(temp)
oneTarg = np.array(['ok',
		'ok',
		'ok',
		'ok',
		'settler',
		'settler',
		'settler',
		'settler',
		'settler',
		'solids',
		'solids',
		'solids',
		'solids'])
oneTestY = np.array([['222','4.5','1518','74','0.25','1642']])

oneDesc[oneDesc == ''] = 0.0
oneDesc = oneDesc.astype(np.float)
oneTestY[oneTestY == ''] = 0.0
oneTestY = oneTestY.astype(np.float)


one_clf = GaussianNB()
one_clf.fit(oneDesc, oneTarg)
oneTestY = one_clf.predict(oneTestY)
print '********** RESULT OF QUESTION ONE **********'
print oneTestY


# *******************************************************************************
# *										*
# *				Question Two 					*
# *										*
# *******************************************************************************

# ***************************************************************
# *	Smooth function			 			*
# ***************************************************************

def smoothListGaussian(list,degree=1):  
	window=degree*2-1  
	weight=np.array([1.0]*window)  
	weightGauss=[]  
	for i in range(window):  
		i=i-degree+1  
		frac=i/float(window)  
		gauss=1/(np.exp((4*(frac))**2))  
		weightGauss.append(gauss)  
	weight=np.array(weightGauss)*weight  
	smoothed=[0.0]*(len(list))  
	for i in range(len(smoothed)):
		if i+window > len(smoothed):
			window = window - 1
			weight=np.array([1.0]*window)  
			weightGauss=[]  
			for j in range(window):  
				j=j-degree+1  
				frac=j/float(window)  
				gauss=1/(np.exp((4*(frac))**2))  
			weightGauss.append(gauss)  
			weight=np.array(weightGauss)*weight 
		smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)  
	return smoothed


# ***************************************************************
# *	Deal on training and testing data 			*
# ***************************************************************

file = open('ML_assignment 3_data.txt', 'r')
originData = file.read()
newData = []
data = []

newData.append(originData.split('\r\n'))
for i in newData:
	for j in i:
		data.append(j.split(','))


temp = []
for i in data:
	if i[0] != '\r':
		temp.append(i)

data = temp
train = []
test = []
temp = 0
for i in data:
	if i[0] != '':
		if i[0][0] == 'D':
			if(temp == 1):
				train.append(i)
			if(temp == 2):
				test.append(i)
		else:
			temp = temp + 1

newtrain = []
newtest = []
for i in train:
	no = 0
	for j in i:
		if j == '?':
			no = 1
	if no == 0:
		newtrain.append(i)

for i in test:
	no = 0
	for j in i:
		if j == '?':
			no = 1
	if no == 0:
		newtest.append(i)

train = np.array(newtrain)
test = np.array(newtest)

# ***************************************************************
# *	Deal on attr data		 			*
# ***************************************************************

fileAttr = open('ML_assignment 3_attr.txt', 'r')
originAttr = fileAttr.read()
newAttr = []
attr = []
newAttr.append(originAttr.split('\r\n'))
for i in newAttr:
	for j in i:
		tt = string.maketrans(' ',',')
		attr.append(j.translate(tt).split(','))

temp = []
get = 0
for i in attr:
	for j in i:
		if 'Cl' in j:
			get = 1
	if get == 1:
		for j in i:
			if 'C' in j or 'D-' in j or j == 'to':
				temp.append(j)


temp2 = []
for i in temp:
	str = ''
	for j in i:
		if j != ' ' and j != '.':
			str += j
	temp2.append(str)


attr = [[],[],[],[],[],[],[]]
classType = 0
for i in temp2:
	if 'C' in i:
		classType = classType + 1
	else:
		attr[classType-1].append(i)

attr = attr[1:6]
for i in attr:
	for j in range(len(i)):
		if i[j] == 'to':
			front = i[j-1]
			back = i[j+1]
			num = 0
			strTemp1 = ''
			strTemp2 = ''
			strSame = ''
			for k in front:
				if k == '-':
					num = 1
				if k == '/':
					num = 2
				if k != '-' and num == 1:
					strTemp1 += k
				if num == 2:
					strSame += k
			num = 0
			for k in back:
				if k == '-':
					num = 1
				if k == '/':
					num = 0
				if k != '-' and num == 1:
					strTemp2 += k
			add = 1
			while(add < (int(strTemp2)-int(strTemp1))):
				strNew = '%d'%(int(strTemp1)+add)
				i.append('D-' + strNew + strSame)
				add = add + 1


for i in attr:
	for j in i:
		if j == 'to':
			i.remove(j)


# *******************************************************************************
# *	Change first column of training and testing data from data to classes	*
# *******************************************************************************

for i in train:
	if i[0] in attr[0]:
		i[0] = '1'
	elif i[0] in attr[1]:
		i[0] = '2'
	elif i[0] in attr[2]:
		i[0] = '3'
	elif i[0] in attr[3]:
		i[0] = '4'
	elif i[0] in attr[4]:
		i[0] = '5'
	else:
		i[0] = '6'

for i in test:
	if i[0] in attr[0]:
		i[0] = '1'
	elif i[0] in attr[1]:
		i[0] = '2'
	elif i[0] in attr[2]:
		i[0] = '3'
	elif i[0] in attr[3]:
		i[0] = '4'
	elif i[0] in attr[4]:
		i[0] = '5'
	else:
		i[0] = '6'

# ***************************************************************
# *	change numpy from string to integer 			*
# ***************************************************************

train[train == ''] = 0.0
train = train.astype(np.float)
test[test == ''] = 0.0
test = test.astype(np.float)
trainX, trainY = train[:, 0],train[:,1:]
testY = test[:,1:]

# ***************************************************************
# *	K-Fold Validation		 			*
# ***************************************************************
'''
knum = 10
kf = StratifiedKFold(n_splits=knum, shuffle = True)
for train, test in kf.split(trainY, trainX):
	y_train, y_test = trainY[train], trainY[test]
	x_train, x_test = trainX[train], trainX[test]

# ***************************************************************
# *	GaussianNB for K-Fold Validation 			*
# ***************************************************************

for num in range(len(y_train[0])):
	y_train[:,num] = np.array(smoothListGaussian(y_train[:,num]));

clf = GaussianNB()
clf.fit(y_train, x_train)
print confusion_matrix(x_test,clf.predict(y_test))
print clf.score(y_test, x_test)
'''
# ***************************************************************
# *	GaussianNB for Real Data	 			*
# ***************************************************************

for num in range(len(trainY[0])):
	trainY[:,num] = np.array(smoothListGaussian(trainY[:,num]));

clf = GaussianNB()
clf.fit(trainY, trainX)
testX = clf.predict(testY)
print '********** RESULT OF QUESTION TWO **********'
print testX

# **********************************************************************


