import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/arish/PycharmProjects/Bafo_Cv/main_funcs/dataset.csv')

X, y = df.iloc[:, 1: ], df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=22)
DT_classifier = DecisionTreeClassifier()
DT_classifier.fit(X_train, y_train)

result = DT_classifier.predict(X_test)
accuracy = accuracy_score(y_test, result)
print(accuracy)


