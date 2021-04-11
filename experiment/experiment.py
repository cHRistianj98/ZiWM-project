from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

dataset_columns = [
    'Age',
    'Sex',
    'On_thyroxine',
    'Query_on_thyroxine',
    'On_antithyroid_medication',
    'Sick',
    'Pregnant',
    'Thyroid_surgery',
    'I131_treatment',
    'Query_hypothyroid',
    'Query_hyperthyroid',
    'Lithium',
    'Goitre',
    'Tumor',
    'Hypopituitary',
    'Psych',
    'TSH',
    'T3',
    'TT4',
    'T4U',
    'FTI'
]

# columns set based on their ranking with indices from thyroid.csv data set
columns_sorted_by_usefulness = {
    'TSH': 16,
    'FTI': 20,
    'TT4': 18,
    'T3': 17,
    'On_thyroxine': 2,
    'Query_hypothyroid': 9,
    'Sex': 1,
    'T4U': 19,
    'Age': 0,
    'Psych': 15,
    'Thyroid_surgery': 7,
    'Pregnant': 6,
    'Sick': 5,
    'Goitre': 12,
    'I131_treatment': 8,
    'Query_on_thyroxine': 3,
    'Lithium': 11,
    'On_antithyroid_medication': 4,
    'Query_hyperthyroid': 10,
    'Tumor': 13,
    'Hypopituitary': 14,
}

# classifiers with manhattan and euclidean metrics
clfs = {
    'KNN2M': KNeighborsClassifier(n_neighbors=2, metric='manhattan'),
    'KNN2E': KNeighborsClassifier(n_neighbors=2, metric='euclidean'),
    'KNN5M': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'KNN5E': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'KNN8M': KNeighborsClassifier(n_neighbors=8, metric='manhattan'),
    'KNN8E': KNeighborsClassifier(n_neighbors=8, metric='euclidean'),
}


# recall_arr = np.concatenate((recall_arr, div[0]), axis=1)
def create_dataset_for_experiment():
    result = np.array(dataset[:, columns_sorted_by_usefulness.get('TSH')]).reshape((dataset.shape[0], 1))
    for key, value in enumerate(columns_sorted_by_usefulness):
        if key == 0:
            continue
        result = np.append(result, np.array(dataset[:, columns_sorted_by_usefulness[value]])
                           .reshape((dataset.shape[0], 1)), axis=1)
    return result


n_datasets = len(columns_sorted_by_usefulness)
n_splits = 5
n_repeats = 2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

# data set from csv file
dataset = np.genfromtxt("../dataset/thyroid.csv", delimiter=", ")
experiment_dataset = create_dataset_for_experiment()
X = experiment_dataset
y = dataset[:, -1].astype(int)
shift = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

for data_id in tqdm(shift):
    if data_id == 0:
        X = experiment_dataset
    else:
        X = experiment_dataset[:, :data_id]
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id + 20, fold_id] = accuracy_score(y[test], y_pred)

mean_scores = np.mean(scores, axis=2).T
# print("\nMean scores:\n", np.array(mean_scores).reshape((126,)))
mean_scores = np.transpose(mean_scores)
pd.DataFrame(mean_scores).to_csv("results.csv", header=None, index=None)