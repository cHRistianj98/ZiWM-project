from scipy.stats import ttest_ind
from tabulate import tabulate
import numpy as np

from ImpKNN import ImpKNN

clfs = {
    'KNN3M': ImpKNN(k=3, metric='manhattan'),
    'KNN3E': ImpKNN(k=3, metric='euclidean'),
    'KNN5M': ImpKNN(k=5, metric='manhattan'),
    'KNN5E': ImpKNN(k=5, metric='euclidean'),
    'KNN7M': ImpKNN(k=7, metric='manhattan'),
    'KNN7E': ImpKNN(k=7, metric='euclidean'),
}

scores_file_name = "stat_tests_data.csv"
stat_better_table_file_name = "stat_better_table_accuracy"
alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

scores = np.genfromtxt(scores_file_name, delimiter=",")
for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

headers = ["KNN3M", "KNN3E", "KNN5M", "KNN5E", "KNN7M", "KNN7E"]
names_column = np.array([["KNN3M"], ["KNN3E"], ["KNN5M"], ["KNN5E"], ["KNN7M"], ["KNN7E"]])
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table, "\n")

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("Advantage:\n", advantage_table, "\n")

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print("Statistical significance (alpha = 0.05):\n", significance_table, "\n")

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("metrics name:", "accuracy", "\nStatistically significantly better:\n", stat_better_table, "\n")

np.save(stat_better_table_file_name, stat_better_table)

results = np.genfromtxt("stat_tests_data.csv", delimiter=",")
results_mean = np.mean(results, axis=1)
results = results_mean.tolist()
np.savetxt("stat_tests.csv", results, delimiter=",")