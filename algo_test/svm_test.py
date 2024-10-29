import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('train_data.csv')
y_train = X_train.iloc[:, 0]
X_train = X_train.drop('label', axis=1)
X_test = pd.read_csv('test_data.csv')
y_test = X_test.iloc[:, 0]
X_test = X_test.drop('label', axis=1)

svm_classifier = SVC(C=1.0, kernel='poly')
svm_classifier.fit(X_train, y_train)
result = svm_classifier.predict()
accuracy = accuracy_score(y_test, result)
print(accuracy)

