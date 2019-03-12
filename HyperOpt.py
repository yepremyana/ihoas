import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from d3m import index
import d3m.metadata.hyperparams as hyperparams
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
import importlib

class JPLHyperOpt(object):
    """
    Wrapped HyperOpt
    """
    def __init__(self, primitive_class, data, target) -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.parameters = {}

    def _enumeration_to_config_space(self, name, hp_value):
        values = hp_value.values
        self.parameters[name] = hp.choice(name, values)
        return

    def _get_hp_search_space(self):
        hyperparameters = self.primitive_class.metadata.query()['primitive_code']['hyperparams']
        configuration = self.primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        for name, description in hyperparameters.items():
            hp_value = configuration[name]
            if description['semantic_types'][0] == 'https://metadata.datadrivendiscovery.org/types/ControlParameter':
                continue
            elif isinstance(hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                self._enumeration_to_config_space(name, hp_value)

        return
            # Define the search space
        #     space = {
        #         'class_weight': hp.choice('class_weight', [None, 'balanced']),
        #         'boosting_type': hp.choice('boosting_type',
        #                                    [{'boosting_type': 'gbdt',
        #                                      'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
        #                                     {'boosting_type': 'dart',
        #                                      'subsample': hp.uniform('dart_subsample', 0.5, 1)},
        #                                     {'boosting_type': 'goss'}]),
        #         'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
        #         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        #         'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
        #         'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        #         'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        #         'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        #         'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
        #     }
    def objective(self, args):

        clf = RandomForestClassifier(**args, random_state=42)

        scores = cross_val_score(clf, self.data, self.target, cv=5)
        return 1 - np.mean(scores)  # Minimize!

        # if 'classification' in self.primitive_class:
        #     cv_iter = StratifiedKFold(n_splits=n_folds,
        #                               shuffle=shuffle,
        #                               random_state=random_state
        #                               ).split(X, y)
        #     for train_index, valid_index in cv_iter:
        #         loss = 1 - accuracy_score(cv_y_pool, cv_pred_pool)
        #
        # else:
        #     cv_iter = KFold(n_splits=n_folds,
        #                     shuffle=shuffle,
        #                     random_state=random_state).split(X)
        #     for train_index, valid_index in cv_iter:
        #         loss = 1 - r2_score(cv_y_pool, cv_pred_pool)

    def optimization(self):
        self._get_hp_search_space()
        # Trials object to track progress
        bayes_trials = Trials()
        MAX_EVALS = 500

        # Optimize
        best = fmin(fn=self.objective, space=self.parameters, algo=tpe.suggest,
                    max_evals=MAX_EVALS, trials=bayes_trials, rstate = np.random.RandomState(52))

        # Sort the trials with lowest loss first
        bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
        bayes_trials_results[:2]
        return

rng = np.random.RandomState(0)
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)
primitive_class = index.get_primitive('d3m.primitives.classification.random_forest.SKlearn')
smac = JPLHyperOpt(primitive_class, iris.data, iris.target)
smac.optimization()