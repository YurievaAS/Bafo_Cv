import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('C:/Users/arish/PycharmProjects/Bafo_Cv/dataset/data_train.csv')
y_train = X_train.iloc[:, 0]
X_train = X_train.drop('label', axis=1)
X_test = pd.read_csv('C:/Users/arish/PycharmProjects/Bafo_Cv/dataset/data_test.csv')
y_test = X_test.iloc[:, 0]
X_test = X_test.drop('label', axis=1)

knn_classifier = KNeighborsClassifier(weights='distance')
knn_classifier.fit(X_train, y_train)

result = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test,result)
print(accuracy)