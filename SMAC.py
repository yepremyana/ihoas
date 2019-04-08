import numpy as np
import pandas as pd
import csv
from timeit import default_timer as timer
import importlib
from pathlib import Path

from sklearn.metrics import accuracy_score, r2_score, precision_score, mean_squared_error,\
                            recall_score, f1_score, explained_variance_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import type_of_target

from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

from config_space import Config_Space

ITERATION = 0

class JPLSMAC(object):
    """
    Wrapped SMAC
    """

    def __init__(self, primitive_class, data, target, dataset_name='',max_evals=50) -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.dataset_name = dataset_name
        self.MAX_EVALS = max_evals
        Path(current_dir + '/Results').mkdir(exist_ok=True, parents=True)
        self.current_dir = current_dir + '/Results'

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        self.import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, self.import_class)
        self.out_file = self.retrieve_path()

    def objective(self, cfg):
        """
        Creates a Primitive based on a configuration and evaluates it on the
        given dataset using cross-validation.
        Parameters:
        -----------
        cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        Returns:
        --------
        A crossvalidated mean score for the svm on the loaded data-set.
        """
        # Keep track of evals
        global ITERATION
        ITERATION += 1

        cfg = {k: cfg[k] for k in cfg}

        # We translate None values:
        for item, key in cfg.items():
            if key == "None":
                cfg[item] = None
        print(cfg)
        cfg = self._translate_union_value(self.union_var, cfg)
        cfg = self._translate_union_value(self.union_choice, cfg)

        start = timer()
        clf = self.sklearn_class(**cfg)
        run_time = timer() - start

        scores = cross_val_score(clf, self.data, self.target, cv=5)
        loss = 1 - np.mean(scores)

        read_connection = open(self.out_file, 'r')
        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        reader = csv.reader(read_connection)
        writer.writerow([loss, cfg, len(list(reader)), run_time])

        return loss # Minimize!

    def _translate_union_value(self, union_list, cfg):
        # We translate Union values:
        for item in union_list:
            if item in cfg:
                value = cfg[item]
                cfg[item] = cfg[value]
                cfg.pop(value, None)  # Remove extra union choices from config
            else:
                continue
        return cfg

    def optimization(self):
        config_space = Config_Space(self.primitive_class)
        self.cs = config_space.get_hp_search_space()
        self.union_var = config_space.get_union_var()
        self.union_choice = config_space.get_union_choice()
        print(self.cs)

        # File to save first results
        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time'])
        of_connection.close()

        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": self.MAX_EVALS,  # maximum function evaluations
                             "cs": self.cs,  # configuration space
                             "deterministic": "false",
                             "memory_limit": 100,
                             })

        # def_value = self.objective(self.cs.get_default_configuration())
        # print("Default Value: %.2f" % (def_value))

        print("Optimizing! Depending on your machine, this might take a few minutes.")
        start = timer()
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=self.objective)
        incumbent = smac.optimize()
        self.best_params = incumbent
        run_time = timer() - start
        self.run_time = run_time
        # inc_value = self.objective(incumbent)
        #inc_value = smac.get_tae_runner().run(incumbent, 1)[1]
        #validate
        # print("Optimized Value: %.2f" % (inc_value))
        print('TIME FOR OPTIMIZATION OVER {} EVALS:'.format(self.MAX_EVALS))
        print(run_time)
        return

    def _save_to_folder(self, path, savefig):
        Path(self.current_dir + path).mkdir(exist_ok=True, parents=True)
        return os.path.join(self.current_dir + path, savefig)

    def retrieve_path(self):
        return self._save_to_folder('/smac_{}_{}'.format(self.import_class, self.dataset_name),
                                    'Hyperparameter_Trials.csv')

    def _classification_scoring(self, test_target, prediction, average_type=None, positive_label=1):
        accuracy = accuracy_score(test_target, prediction)
        f1 = f1_score(test_target, prediction, average=average_type, pos_label=positive_label)
        precision = precision_score(test_target, prediction, average=average_type, pos_label=positive_label)
        recall = recall_score(test_target, prediction, average=average_type, pos_label=positive_label)

        return {'optimization_technique': 'smac', 'estimator': str(self.primitive_class),
                'dataset': self.dataset_name, 'accuracy_score': accuracy, 'precision_score': precision,
                'recall_score': recall, 'f1_score': f1,
                'max_evals': self.MAX_EVALS, 'total_time': self.run_time,
                'best_params': self.best_params}

    def validate(self, test_data, test_target):
        cfg = {k: self.best_params[k] for k in self.best_params}

        # We translate None values:
        for item, key in cfg.items():
            if key == "None":
                cfg[item] = None

        cfg = self._translate_union_value(self.union_var, cfg)
        cfg = self._translate_union_value(self.union_choice, cfg)

        best_model = self.sklearn_class(**cfg)
        best_model.fit(self.data, self.target)
        prediction = best_model.predict(test_data)
        score = best_model.score(test_data, test_target)

        if 'classification' in str(self.primitive_class):
            type_target = type_of_target(test_target)

            if type_target == "binary":
                series_target = pd.Series(test_target)
                positive_label = series_target.value_counts().index[1]
                scores_dict = self._classification_scoring(test_target,
                                                           prediction,
                                                           average_type='binary',
                                                           positive_label=positive_label)
                roc_auc = roc_auc_score(test_target, prediction)
                scores_dict['roc_auc'] = roc_auc
                scores_dict['score'] = score

            elif type_target == "multiclass":
                scores_dict = self._classification_scoring(test_target,
                                                           prediction,
                                                           average_type='macro')
                scores_dict['score'] = score

        elif 'regression' in str(self.primitive_class):
            r2 = r2_score(test_target, prediction)
            mse = mean_squared_error(test_target, prediction)
            explained_variance = explained_variance_score(test_target, prediction)
            scores_dict = {'optimization_technique': 'smac', 'estimator': str(self.primitive_class),
                           'dataset': self.dataset_name, 'r2': r2, 'explained_variance_score': explained_variance,
                           'mean_squared_error': mse, 'max_evals': self.MAX_EVALS,
                           'total_time':self.run_time,'best_params':self.best_params,'score': score}

        return scores_dict

# need to make a utils file with basic things list lower, upper, save to folder, retrieve path
# need to now work with the validation code that they have and whatever else
# need to check the other functions SMAC has
# need to fix the default in union, it is wrong bc i am not setting the choices default but configuration wont give it to me, self.default_hyperparameter configuration[default]
# understand why global iteration wont work