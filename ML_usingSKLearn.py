#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

traindata = 'traindata.txt'
data = np.loadtxt(traindata, dtype = str)
new_data = []
for i in range(len(data)):
    data_row = []
    inte = ""
    for j in data[i]:
        if j == ',':
            data_row.append(float(inte))
            inte = ''
        else:
            inte = inte + j
    new_data.append(data_row)
data = np.array(new_data)
# print(data)


# In[162]:


tranlabel = 'trainlabels.txt'
labels = np.loadtxt(tranlabel, dtype = int)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1, random_state=0)

testdata = 'testdata.txt'
testdata = np.loadtxt(testdata, dtype = str)
new_data = []
for i in range(len(testdata)):
    data_row = []
    inte = ""
    for j in testdata[i]:
        if j == ',':
            data_row.append(float(inte))
            inte = ''
        else:
            inte = inte + j
    new_data.append(data_row)
testdata = np.array(new_data)
# print(data)
tra = 'testlabels (copy).txt'
l = np.loadtxt(tra, dtype = int)
# In[164]:


# k is the number of features to use
k = 26

rfc = RandomForestClassifier()

rfc.fit(train_data, train_labels)

# Get feature importances
importances = rfc.feature_importances_
# print(importances)
# print(train_data[0])
# Select top k features based on importance scores
modified_rfc = np.argsort(importances)[::-1]

# print(modified_rfc)
selected_rfc = modified_rfc[:k]
# print(selected_rfc)
train_data = train_data[:, selected_rfc]
test_data = test_data[:,selected_rfc]
testdata = testdata[:,selected_rfc]
# #print(train_data)

rfc.fit(train_data, train_labels)

test_prediction = rfc.predict(test_data)

# print(test_prediction)
test = rfc.predict(testdata)
# print(test_labels)
with open('testlabels.txt', 'w') as f:
    for i in test_prediction:
        f.write(str(i) + "\n")
f.close()

accuracy = accuracy_score(test_labels, test_prediction)
print(accuracy)
classification_report = classification_report(test_labels, test_prediction)
print(classification_report)
accuracyt = accuracy_score(l, test)
print(accuracyt)
# classification_reportt = classification_report(l, test)
# print(classification_reportt)





