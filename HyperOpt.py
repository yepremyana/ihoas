import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm.classes import SVR
from d3m import index
import d3m.metadata.hyperparams as hyperparams
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt.pyll import scope
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
        self.choice_names = []

    def _enumeration_to_config_space(self, name, hp_value):
        values = hp_value.values
        params_config = hp.choice(name, values)
        return params_config

    def _constant_to_config_space(self, name, hp_value):
        default_hp = hp_value.get_default()
        return default_hp

    def _bounded_to_config_space(self, name, hp_value):
        lower, default_hp = hp_value.lower, hp_value.get_default()
        if hp_value.upper == None:
            if default_hp == 0:
                upper = 10
            else:
                upper = 2 * (default_hp)
        else:
            upper = hp_value.upper
        structure_type = hp_value.structural_type

        if issubclass(structure_type, float):
            params_config = hp.uniform(name, lower, upper)

        elif issubclass(structure_type, int):
            params_config = scope.int(hp.quniform(name, lower, upper, 1))

        return params_config

    def _union_to_config_space(self, name, hp_value):
        values_union = []
        for union_name, union_hp_value in hp_value.configuration.items():
            label = "{}_{}".format(name, union_name)
            if isinstance(union_hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                child = self._bounded_to_config_space(label, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                child = self._enumeration_to_config_space(label, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Constant)):
                child = self._constant_to_config_space(label, union_hp_value)
            values_union.append(child)
        params_config = hp.choice(name, values_union)
        return params_config

    def _choice_to_config_space(self, name, hp_value):
        choice_combo = []
        for choice, hyperparameter in hp_value.choices.items():
            choice_dict = {}
            choice_dict[name] = choice
            for type, hp_info in hyperparameter.configuration.items():
                if type != 'choice':
                    label = "{}_{}".format(choice, type)
                    if isinstance(hp_info, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                        values_union = self._bounded_to_config_space(label, hp_info)
                        choice_dict[type] = values_union
                    elif isinstance(hp_info, (hyperparams.Constant)):
                        values_union = self._constant_to_config_space(label, hp_info)
                        choice_dict[type] = values_union
                    elif isinstance(hp_info, (hyperparams.Union)):
                        values_union = self._union_to_config_space(label, hp_info)
                        choice_dict[type] = values_union
            choice_combo.append(choice_dict)

        print(choice_combo)
        return choice_combo

    def _get_hp_search_space(self):
        hyperparameters = self.primitive_class.metadata.query()['primitive_code']['hyperparams']
        configuration = self.primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        for name, description in hyperparameters.items():
            hp_value = configuration[name]
            if description['semantic_types'][0] == 'https://metadata.datadrivendiscovery.org/types/ControlParameter':
                continue
            elif isinstance(hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                params_config = self._enumeration_to_config_space(name, hp_value)
                self.parameters[name] = params_config
            elif isinstance(hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                params_config = self._bounded_to_config_space(name, hp_value)
                self.parameters[name] = params_config
            elif isinstance(hp_value, (hyperparams.Union)):
                params_config = self._union_to_config_space(name, hp_value)
                self.parameters[name] = params_config
            elif isinstance(hp_value, (hyperparams.Choice)):
                params_config = self._choice_to_config_space(name, hp_value)
                self.parameters[name] = hp.choice(name, params_config)
                self.choice_names.append(name)
            elif isinstance(hp_value, (hyperparams.Constant)):
                params_config = self._constant_to_config_space(name, hp_value)
                self.parameters[name] = params_config
        return

    def objective(self, args):

        args = self._translate_union_value(self.choice_names, args)
        print(args)
        clf = RandomForestClassifier(**args, random_state=42)

        # clf = SVR(**args)
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

    def _translate_union_value(self, choice_list, args):
        # We translate Choice values:
        for item in choice_list:
            for key in args[item]:
                if key != item:
                    args[key] = args[item][key]
                else:
                    continue
            args[item] = args[item][item]
        return args

    def optimization(self):
        self._get_hp_search_space()
        # Trials object to track progress
        bayes_trials = Trials()
        MAX_EVALS = 15
        # print(self.parameters)
        # print(sample(self.parameters))

        # Optimize
        best = fmin(fn=self.objective, space=self.parameters, algo=tpe.suggest,
                    max_evals=MAX_EVALS, trials=bayes_trials, rstate = np.random.RandomState(52))
        print(best)
        # Sort the trials with lowest loss first
        bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
        print(bayes_trials_results[:2])
        return

rng = np.random.RandomState(0)
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)
primitive_class = index.get_primitive('d3m.primitives.classification.random_forest.SKlearn')
# primitive_class = index.get_primitive('d3m.primitives.regression.svr.SKlearn')
smac = JPLHyperOpt(primitive_class, iris.data, iris.target)
smac.optimization()
