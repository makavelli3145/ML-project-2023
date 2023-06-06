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

def get_probabilities(train_labels):
    probability = np.zeros(10)
    labels = np.zeros(10)
    for each_label in train_labels:
        for index in range(len(probability)):
            if each_label == index:
                probability[index] = probability[index] + 1
                labels[index] = labels[index] + 1
    for each_label in range(len(probability)):
        probability[each_label] = probability[each_label] / len(train_labels)
    return np.around(probability, 5), labels

def meanVareience(train_data, train_labels, label, number, feature_index):
    mean = 0
    varience = 0
    for index in range(len(train_labels)):
        if train_labels[index] == label:
            mean = mean + train_data[index][feature_index]
    mean = mean / number
    for index in range(len(train_labels)):
        if train_labels[index] == label:
            varience = varience + ((train_data[index][feature_index] - mean)  ** 2)
    varience = varience / number
    return mean, varience
            

def prob_given(train_data, train_labels, label, number, feature_index, predicting_row):
    mean, varience = meanVareience(train_data, train_labels, label, number, feature_index)
    first = (1 / (np.sqrt((2 * np.pi) * varience)))
    exp = (-(1 / 2) * (((predicting_row[feature_index] - mean) ** 2) / varience))
    prob_label = first * exp
    return prob_label

def prob_given_feature(train_data, train_labels, label, number, predicting_row):
    prob_label = 1
    for each_feature in range(len(predicting_row)):
        prob_label = prob_label * prob_given(train_data, train_labels, label, number, each_feature, predicting_row)
    return prob_label

def prob_being_label(train_data, train_labels, predicting_row):
    prob_label, label_size = get_probabilities(train_labels)
    prob_for_each = np.zeros(10)
    for each_label in range(len(label_size)):
        prob_for_each[each_label] = (prob_given_feature(train_data, train_labels, each_label, label_size[each_label], predicting_row) * prob_label[each_label])
    x = sum(prob_for_each)
    for each_label in range(len(label_size)):
        prob_for_each[each_label] = prob_for_each[each_label] / x
    return prob_for_each, max(prob_for_each)



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

tranlabel = 'trainlabels.txt'
labels = np.loadtxt(tranlabel, dtype = int)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=0)
print(prob_being_label(train_data, train_labels, test_data[0]))
print(prob_being_label(train_data, train_labels, test_data[1]))
print(test_labels[0])
print(test_labels[1])
print(prob_being_label(train_data, train_labels, test_data[2]))
print(test_labels[2])
print(prob_being_label(train_data, train_labels, test_data[3]))
print(test_labels[3])
