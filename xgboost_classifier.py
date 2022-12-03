import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from scipy.io import mmread
import csv

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import gzip
import json
param_grid = {
    "max_depth": [10, 15, 30],
    # 50, 100
}


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


def compress(jsonfilename):
    data = json.load(open(jsonfilename))
    with gzip.open(f"{jsonfilename}.gzip", 'w') as fout:
        fout.write(json.dumps(data).encode('utf-8'))


def decompress(jsonfilename):
    with gzip.open(jsonfilename, 'r') as fin:
        data = json.loads(fin.read().decode('utf-8'))


if __name__ == "__main__":

    xgb_cl = xgb.XGBClassifier(use_label_encoder=False, max_depth=20, n_estimators=500, tree_method='gpu_hist')
    # clf = SVC(gamma=20)

    # {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 7, 'reg_lambda': 0, 'scale_pos_weight': 3, 'subsample': 0.8}
    # scale_pos_weight handles imbalances: sum(negative instances) / sum(positive instances)
    # xgb_cl = xgb.XGBClassifier(colsample_bytree=0.5, gamma=0, n_estimators=500,
    #                            learning_rate=0.3, max_depth=20, reg_lambda=0, scale_pos_weight=1, subsample=0.8)
    # tree_method='gpu_hist',
    print(xgb_cl.get_params())

    edges, labels = get_data("data/graph-1000-0.501-small-world-p-0.5.mtx")
    xgb_cl.fit(edges, labels)
    preds = xgb_cl.predict(edges)
    print(accuracy_score(labels, preds))

    # xgb_cl.save_model("model.json")
    #     # xgb_cl = xgb.XGBClassifier(
    # compress("model.json")
    #     objective="binary:logistic", tree_method="gpu_hist", use_label_encoder=False)

    # Init Grid Search
    # grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=1,
    #                        cv=3, scoring="roc_auc")

    # _ = grid_cv.fit(edges, labels)

    # print(grid_cv.best_score_)
    # print(grid_cv.best_params_)

    # clf.fit(edges, labels)

    # # predicted = clf.predict([[0, 2], [0, 10]])
    # # print(predicted)
    # predicted = clf.predict(edges)
    # print(accuracy_score(labels, predicted))

    # clf = clf.fit(edges, labels)
    # scores = cross_val_score(clf, edges, labels, cv=5)
    # print(scores.mean())
