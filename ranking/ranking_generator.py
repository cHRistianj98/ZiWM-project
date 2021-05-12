import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.feature_selection import SelectKBest, chi2, f_classif


def build_features_ranking(x, y, score_func):
    features_num = x.shape[1]
    k_best_selector = SelectKBest(score_func=score_func, k=features_num)
    k_best_selector.fit(x, y)
    scores_ranking = [
        (name, round(score, 2))
        for name, score in zip(dataset_columns, k_best_selector.scores_)
    ]
    scores_ranking.sort(reverse=True, key=lambda x: x[1])
    return scores_ranking


def print_features_ranking_with_plot(features_ranking, used_score_func):
    print(f'Features ranking after using {used_score_func} score function:')
    for i, feature in enumerate(features_ranking, 1):
        print(f"{i}. {feature[0]} {feature[1]}")
    # display bar plot
    plt.figure(figsize=(10, 100))
    estimator_num = len(features_ranking)
    ascending_features = sorted([(f[0], f[1]) for f in features_ranking], key=lambda f: f[1])
    plt.barh(range(estimator_num), [feature[1] for feature in ascending_features],
             align='center')  # extract score value
    plt.yticks(range(estimator_num), [feature[0] for feature in ascending_features])
    plt.title(f'Ranking oparty na {used_score_func}')
    plt.xscale('log')
    plt.show()
    plt.savefig(fname='ranking.png', orientation='landscape')


dataset_classes = {
    1: "Normal",
    2: "Hyperthyroidism",
    3: "Hypothyroidism",
}

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
sns.set_theme()
data_list = []
scores = "thyroid.csv"
for i, file in enumerate(glob.glob("../dataset/thyroid.csv"), 1):
    data_set = pd.read_csv(file, sep=", ", engine='python', header=None)
    data_list.append(data_set)

dataset = pd.concat(data_list, axis=0)
dataset.info()

x = dataset.drop(21, axis=1)
y = dataset[21]
# f_classif: ANOVA test (F-value between label/feature for regression tasks)
features_ranking_classif = build_features_ranking(x, y, f_classif)
print_features_ranking_with_plot(features_ranking_classif, 'f_classif')
# chi-squared stats of non-negative features for classification tasks.
# features_ranking_chi = build_features_ranking(x, y, chi2)
# print_features_ranking_with_plot(features_ranking_chi, 'chi2')
