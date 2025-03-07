import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection


DATASET_PATH = "dataset/iris.csv"  # 본인이 iris.csv를 저장한 경로를 입력합니다.

if __name__ == "__main__":
    iris_dataset = pd.read_csv(DATASET_PATH)
    iris_dataset['target'] = iris_dataset['Name'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

    x_data = iris_dataset.iloc[:, :4]
    y_data = iris_dataset.iloc[:, [-1]]

    classifier_list = []
    classifier_list.append(("DT", DecisionTreeClassifier()))
    classifier_list.append(("RF", SVC()))
    classifier_list.append(("SVM", RandomForestClassifier()))
    classifier_list.append(("LR", LogisticRegression()))
    
    results = []
    names = []
    
    for name, classifier in classifier_list:
        kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = model_selection.cross_val_score(classifier, x_data, y_data.values.ravel(), cv=kfold, scoring="accuracy")

        print(name, ":", cv_results)
        results.append(cv_results)
        names.append(name)
            
    fig = plt.figure()

    fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()