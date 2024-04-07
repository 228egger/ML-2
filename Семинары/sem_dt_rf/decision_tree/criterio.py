from typing import Tuple, Union

import numpy as np


class Criterion:
    def get_best_split(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """
        Parameters
        ----------
        feature : feature vector, np.ndarray.shape = (n_samples, )
        target  : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        threshold : value to split feature vector, float
        q_value   : impurity improvement, float
        """
        ind = feature.argsort()
        target_sorted = target[ind]

        q_max = -np.inf
        i_max = None
        q_all = self.score(target)
        for i in range(feature.shape[0]):
            q_left = self.score(target_sorted[:i])
            q_right = self.score(target_sorted[i:])
            q = q_all - i / feature.shape[0] * q_left - (feature.shape[0] - i) / feature.shape[0] * q_right
            if q > q_max:
                i_max, q_max = i, q

        threshold = (feature[ind[i_max]] + feature[ind[i_max] - 1]) / 2
        return threshold, q_max

    def score(self, target: np.ndarray) -> float:
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        impurity : float
        """

        raise NotImplementedError

    def get_predict_val(self, target: np.ndarray) -> Union[float, np.ndarray]:
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        prediction :
            - classification: probability distribution in node, np.ndarray.shape = (n_classes, )
            - regression: best constant approximation, float
        """

        raise NotImplementedError

class GiniCriterion(Criterion):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def get_predict_val(self, classes: np.ndarray) -> np.ndarray:
        pred = np.bincount(classes, minlength=self.n_classes) / classes.shape[0]
        return pred

    def score(self, classes):
        pred = self.get_predict_val(classes)
        return 1 - (pred ** 2).sum()


class EntropyCriterion(Criterion):
    EPS = 1e-6

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def get_predict_val(self, classes):
        pred = np.bincount(classes, minlength=self.n_classes) / classes.shape[0]
        return pred

    def score(self, classes):
        pred = self.get_predict_val(classes)
        return -np.sum(pred * np.log(pred + self.EPS))


class MSECriterion(Criterion):

    def __init__(self):
        pass

    def get_predict_val(self, target):
        return np.mean(target)
        
        
    def score(self, target):
        pred = self.get_predict_val(target)
        return np.mean((target - pred) ** 2)
