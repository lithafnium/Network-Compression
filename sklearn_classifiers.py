import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from scipy.io import mmread
import csv


def get_data(path):
    a = mmread(path)
    labels = []
    edges = []

    for i in range(len(a)):
        for j in range(len(a)):
            edges.append([i, j])
            if a[i][j] == 1:
                labels.append(1)
            else:
                labels.append(0)
    # with open(f"graph.csv", "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id_1", "id_2", "label"])
    #     for i, edge in enumerate(edges):
    #         writer.writerow([edge[0], edge[1], labels[i]])
    return edges, labels


if __name__ == "__main__":
    clf = SVC(gamma=20)

    # clf = RandomForestClassifier(
    #     n_estimators=10, max_depth=None, min_samples_split=2, random_state=0
    # )

    edges, labels = get_data("data/graph-100-0.040-small-world-p-0.5.mtx")
    clf.fit(edges, labels)

    # predicted = clf.predict([[0, 2], [0, 10]])
    # print(predicted)
    predicted = clf.predict(edges)
    print(accuracy_score(labels, predicted))

    # # clf = clf.fit(edges, labels)
    # scores = cross_val_score(clf, edges, labels, cv=5)
    # print(scores.mean())
