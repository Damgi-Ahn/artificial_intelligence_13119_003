import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET_PATH = "dataset/iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력합니다.

if __name__ == "__main__":
    iris_dataset = pd.read_csv(DATASET_PATH)
    # iris_dataset['target'] = iris_dataset['Name'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

    x_data = iris_dataset.iloc[:, :-1]
    y_data = iris_dataset.iloc[:, -1] 

    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)
    
    print("X_train: ", X_train)
    print("X_test: ", X_test)
    print("Y_train: ", Y_train)
    print("Y_test: ", Y_test)
    
    DT = DecisionTreeClassifier()
    DT.fit(X_train, Y_train)
    dt_pred = DT.predict(X_test)
    print("DecisionTreeClassifier: ", accuracy_score(Y_test, dt_pred))
    
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    rf_pred = rf.predict(X_test)
    print("RandomForestClassifier: ", accuracy_score(Y_test, rf_pred))
    
    svm = SVC()
    svm.fit(X_train, Y_train)
    svm_pred = svm.predict(X_test)
    print("SVM: ", accuracy_score(Y_test, svm_pred))

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train, Y_train)
    lr_pred = lr.predict(X_test)
    print("LogisticRegression: ", accuracy_score(Y_test, lr_pred))

