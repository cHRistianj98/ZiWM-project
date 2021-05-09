import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import DistanceMetric
from scipy import stats
from tqdm import tqdm


class ImpKNN(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------

     k : int (Default = 7)
        Number of neighbors used to predict class of new sample.

     metric : (Deafult = 'euclidean')
        Metric to calculate distance between samples.

    """

    def __init__(self, k=7, metric='euclidean'):
        self.metric = metric
        self.k = k

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y

        return self

    def get_neighbors(self, x_query, Validation_set):
        neighbors = []

        for val_point in Validation_set:
            dist = DistanceMetric.get_metric(self.metric)
            points = [x_query, val_point[0]]
            result = np.array(dist.pairwise(points)).max()
            neighbors.append([val_point, result])

        neighbors = sorted(neighbors, key=lambda x: x[1])[:self.k]
        neighbors = [n[0] for n in neighbors]

        return neighbors

    def predict(self, X):
        y_pred = []
        for x_query in tqdm(X):
            neighbors = self.get_neighbors(x_query, zip(self.X_, self.y_))
            neighbors_labels = []
            for k in range(self.k):
                neighbors_labels.append(neighbors[k][1])
            mode, _ = stats.mode(neighbors_labels)
            y_pred.append(mode[0])
        return y_pred