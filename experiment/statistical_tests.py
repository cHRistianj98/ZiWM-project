from scipy.stats import ttest_ind
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import numpy as np

clfs = {
    'KNN2M': KNeighborsClassifier(n_neighbors=2, metric='manhattan'),
    'KNN2E': KNeighborsClassifier(n_neighbors=2, metric='euclidean'),
    'KNN5M': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
    'KNN5E': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'KNN8M': KNeighborsClassifier(n_neighbors=8, metric='manhattan'),
    'KNN8E': KNeighborsClassifier(n_neighbors=8, metric='euclidean'),
}

scores_file_name = "results.csv"
stat_better_table_file_name = "stat_better_table_precision"
alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

scores = np.genfromtxt(scores_file_name, delimiter=",")
for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

headers = ["KNN2M", "KNN2E", "KNN5M", "KNN5E", "KNN8M", "KNN8E"]
names_column = np.array([["KNN2M"], ["KNN2E"], ["KNN5M"], ["KNN5E"], ["KNN8M"], ["KNN8E"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table, "\n")

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table, "\n")

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table, "\n")

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
# print("metrics name:", "precision", "\nStatistically significantly better:\n", stat_better_table, "\n")

np.save(stat_better_table_file_name, stat_better_table)

# create results.csv
results = np.genfromtxt("results.csv", delimiter=",")
results_mean = np.mean(results, axis=1)
results = results_mean.tolist()
np.savetxt("stat_tests.csv", results, delimiter=",")










