from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class EPMRandomForest(ForestRegressor):
    def __init__(
            self, 
            n_estimators: int = 100, 
            *, 
            log=False,
            cross_trees_variance=False,
            criterion="squared_error",
            splitter="random",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap: bool = False, 
            oob_score: bool = False, 
            n_jobs=None, 
            random_state=None, 
            verbose: int = 0, 
            warm_start: bool = False, 
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None) -> None:
        super().__init__(
            DecisionTreeRegressor(), 
            n_estimators, 
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ), 
            bootstrap=bootstrap, 
            oob_score=oob_score, 
            n_jobs=n_jobs, 
            random_state=random_state, 
            verbose=verbose, 
            warm_start=warm_start, 
            max_samples=max_samples)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.splitter = splitter
        self.log = log
        self.cross_trees_variance = cross_trees_variance
    
    def fit(self, X, y, sample_weight=None):
        assert sample_weight is None, "Sample weights are not supported"
        super().fit(X=X, y=y, sample_weight=sample_weight)

        
        self.trainX = X
        self.trainY = y
        if self.log:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(self, X):
        preds = []
        for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
            preds.append(tree.predict(X))
        preds = np.array(preds).T
        
        means = preds.mean(axis=1)
        vars = preds.var(axis=1)

        return means.reshape(-1, 1), vars.reshape(-1, 1)


