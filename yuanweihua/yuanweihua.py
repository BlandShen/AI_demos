import numpy as np
from matplotlib import colors
from sklearn import svm 
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl

def iris_type(s):
    it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
    return it[s]

data_path = '/home/bland/note/demo/data/iris.data'
data = np.loadtxt(data_path, dtype = float, delimiter = ',' , converters = {4: iris_type})
#x,y = np.split(data, (4,), axis = 1)
x = data[:, 0:2]#第一维（行）全取，第二维（列）取index为0，1（前两列）
y = data[:, 4:]#第一维（行）全取，第二维（列）取index为4（第五列）
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state = 1, test_size = 0.3)

def classifier():
    clf = svm.SVC(C =0.8, kernel = 'rbf', gamma = 50,decision_function_shape = 'ovr')
    return clf

clf = classifier()

def train(clf, x_train, y_train):
    clf.fit(x_train, y_train.ravel())

train(clf,x_train,y_train)

def show_accuracy(a, b, tip):#这个函数是不是把score的实现重写了一遍？
    acc =a.ravel() == b.ravel()
    print('{0:} 准确率:{1:.3f}' .format(tip, np.mean(acc)))

def print_accuracy(clf,x_train,y_train,x_test,y_test):
    print("训练集准确率为{:.3f}".format(clf.score(x_train, y_train)))#返回给定测试数据和标签的平均精确度
    print("测试集准确率为{:.3f}".format(clf.score(x_test, y_test)))

    show_accuracy(clf.predict(x_train), y_train, '训练数据')#使用模型后的结果
    show_accuracy(clf.predict(x_test), y_test, '测试数据')#使用模型后的结果
    #print('decision_function:\n', clf.decision_function(x_train))
print_accuracy(clf, x_train, y_train, x_test, y_test)