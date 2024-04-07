from typing import Tuple, Optional
import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion


class TreeNode:
    def __init__(self, impurity: float, predict_val: np.ndarray, depth: int):
        self.impurity = impurity  # node impurity
        self.predict_val = predict_val  # prediction of node
        self.depth = depth  # current node depth

        self.feature = None  # feature to split
        self.threshold = None  # threshold to split
        self.improvement = -np.inf  # node impurity improvement after split

        self.child_left = None
        self.child_right = None

    @property
    def is_terminal(self) -> bool:
        return self.child_left is None and self.child_right is None

    @classmethod
    def get_best_split(cls, x: np.ndarray, y: np.ndarray, criterion: Criterion) -> Tuple[float, float, int]:
        """
        Finds best split for current node

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion: criterion

        Returns
        -------
        q_value   : impurity improvement,   float
        threshold : value to split feature, float
        feature   : best feature to split,  int
        """
        q_max, threshold_max, feature_max = -np.inf, None, None
        for fi in range(x.shape[1]):
            threshold, q = criterion.get_best_split(x[:, fi], y)
            if q > q_max:
                q_max, threshold_max, feature_max = q, threshold, fi

        return q_max, threshold_max, feature_max

    def get_best_split_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)

        Returns
        -------
        right_mask : indicates samples in right node after split
            np.ndarray.shape = (n_samples, )
            np.ndarray.dtype = bool
        """
        return np.asarray(x[:, self.feature] >= self.threshold)

    def split(self, x: np.ndarray, y: np.ndarray, criterion: Criterion, feature, threshold, improvement):
        """
        Split current node

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion : criterion to split by, Criterion
        improvement   : impurity improvement,   float
        threshold : value to split feature, float
        feature   : best feature to split,  int

        Returns
        -------
        right_mask : indicates samples in right node after split
            np.ndarray.shape = (n_samples, )
            np.ndarray.dtype = bool

        child_left  : TreeNode
        child_right : TreeNode
        """
        self.feature = feature
        self.threshold = threshold
        self.improvement = improvement
        mask = self.get_best_split_mask(x)

        self.child_left = TreeNode(
            depth=self.depth + 1,
            predict_val=criterion.get_predict_val(y[~mask]),
            impurity=criterion.score(y[~mask])
        )
        self.child_right = TreeNode(
            depth=self.depth + 1,
            predict_val=criterion.get_predict_val(y[mask]),
            impurity=criterion.score(y[mask])
        )
        return mask, self.child_left, self.child_right
