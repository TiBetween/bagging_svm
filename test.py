from pandas.io import pickle
import pickle

import load_dataset as ld
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import cv2 as cv
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

dir_path = "./cifar-10-batches-py/"


def load_train_dataset():
    x_r, x_g, x_b, y, filenames = ld.load_dataset_with_pic_tune(dir_path + "data_batch_1")
    for i in range(2, 6):
        temp_r, temp_g, temp_b, temp_y, temp_filenames = ld.load_dataset_with_pic_tune(dir_path + f"data_batch_{i}")
        x_r.extend(temp_r)
        x_g.extend(temp_g)
        x_b.extend(temp_b)
        y.extend(temp_y)
        filenames.extend(temp_filenames)
    return x_r, x_g, x_b, y, filenames


def load_test_dataset():
    x_r, x_g, x_b, y, filenames = ld.load_dataset_with_pic_tune(dir_path + "test_batch")
    return x_r, x_g, x_b, y, filenames


def preprocess(list):
    length = len(list)
    for i in range(length):
        list[i] = list[i].reshape(-1, 1)
        std = StandardScaler()
        std.fit(list[i])
        list[i] = std.transform(list[i]).ravel()
    return list


M_r = []
M_g = []
M_b = []
for i in range(1, 6):
    model = SVC(kernel="rbf", C=5, verbose=1)
    x_train_r, x_train_g, x_train_b, y_train, filename_train = ld.load_dataset_with_pic_tune(
        dir_path + f"data_batch_{i}")
    x_train_r = preprocess(x_train_r)
    x_train_g = preprocess(x_train_g)
    x_train_b = preprocess(x_train_b)

    model.fit(x_train_r, y_train)
    M_r.append(model)
    with open(f'./models/linear_r{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
        print('dump success')
    model.fit(x_train_g, y_train)
    M_g.append(model)

    with open(f'./models/linear_g{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
        print('dump success')
    model.fit(x_train_b, y_train)
    M_b.append(model)

    with open(f'./models/linear_b{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
        print('dump success')

x_test_r, x_test_g, x_test_b, y_test, filename_test = load_test_dataset()
x_test_r = preprocess(x_test_r)
x_test_g = preprocess(x_test_g)
x_test_b = preprocess(x_test_b)

answer = []
for i in M_r:
    y_ = i.predict(x_test_r)
    answer.append(y_)
for i in M_g:
    y_ = i.predict(x_test_r)
    answer.append(y_)
for i in M_b:
    y_ = i.predict(x_test_r)
    answer.append(y_)

answer = np.array(answer)
y_ = np.zeros(len(x_test_r), dtype=np.int)
for i in range(len(x_test_r)):
    ans = answer[:, i:i + 1].ravel()
    counts = np.bincount(ans)
    temp = np.argmax(counts)
    y_[i] = temp

print(confusion_matrix(y_test, y_))
print(accuracy_score(y_test, y_))
