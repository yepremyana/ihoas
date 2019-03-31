import numpy as np
from sklearn.metrics import accuracy_score, r2_score, precision_score, \
                            recall_score, f1_score, explained_variance_score
from sklearn.model_selection import cross_val_score
import d3m.metadata.hyperparams as hyperparams
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.pyll import scope
import importlib
import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path
import csv
import json
from timeit import default_timer as timer
from hyperopt import space_eval

ITERATION = 0

class JPLHyperOpt(object):
    """
    Wrapped HyperOpt
    """

    def __init__(self, primitive_class, data, target, dataset_name='',max_evals=50) -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.dataset_name = dataset_name
        self.parameters = {}
        self.choice_names = []
        self.best_params = None
        self.MAX_EVALS = max_evals
        self.run_time = None
        Path(current_dir + '/Results').mkdir(exist_ok=True, parents=True)
        self.current_dir = current_dir + '/Results'

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        self.import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, self.import_class)
        self.out_file = self.retrieve_path()

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
                upper = 2 * (abs(default_hp))
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
        # Keep track of evals
        global ITERATION
        ITERATION += 1

        args = self._translate_union_value(self.choice_names, args)
        print(args)

        start = timer()
        clf = self.sklearn_class(**args)
        run_time = timer() - start

        scores = cross_val_score(clf, self.data, self.target, cv=5)
        loss = 1 - np.mean(scores)

        of_connection = open(self.out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([loss, args, ITERATION, run_time])
        of_connection.close()

        return loss  # Minimize!

    def _save_to_folder(self, path, savefig):
        Path(self.current_dir + path).mkdir(exist_ok=True, parents=True)
        return os.path.join(self.current_dir + path, savefig)

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

        # File to save first results
        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time'])
        of_connection.close()

        # Optimize
        start = timer()
        best = fmin(fn=self.objective, space=self.parameters, algo=tpe.suggest,
                    max_evals=self.MAX_EVALS, trials=bayes_trials, rstate = np.random.RandomState(52))
        run_time = timer() - start
        self.run_time = run_time
        # Sort the trials with lowest loss first
        self.save_trials(bayes_trials.results)
        bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
        self.best_params = space_eval(self.parameters, best)
        print('The parameters combination that would give best accuracy is : ')
        print(best)
        print('The min loss achieved after parameter tuning via hyperopt is : ', bayes_trials_results[:1][0]['loss'])
        # print(bayes_trials_results[:1][0]['loss'])
        # ToDO: record stats
        #make a new with the stats and csv and anything else, then add the figures to the folder.
        print('TIME FOR OPTIMIZATION OVER {} EVALS:'.format(self.MAX_EVALS))
        print(run_time)
        return self.best_params

    def retrieve_path(self):
        return self._save_to_folder('/hyperopt_{}_{}'.format(self.import_class, self.dataset_name), 'Hyperparameter_Trials.csv')

    def validate(self, test_data, test_target, pos_label, average):
        best_model = self.sklearn_class(**self.best_params)
        best_model.fit(self.data, self.target)
        prediction = best_model.predict(test_data)
        score = best_model.score(test_data, test_target)

        if 'classification' in str(self.primitive_class):
            f1 = f1_score(test_target, prediction, average=average, pos_label=pos_label)
            accuracy = accuracy_score(test_target, prediction)
            precision = precision_score(test_target, prediction, average=average, pos_label=pos_label)
            recall = recall_score(test_target, prediction, average=average, pos_label=pos_label)
            #confusion matrix
            scores_dict = {'optimization_technique': 'hyperopt', 'estimator': str(self.primitive_class), 'dataset': self.dataset_name, 'accuracy_score': accuracy, 'precision_score': precision, 'recall_score': recall, 'f1_score': f1, 'prediction': prediction, 'score': score, 'max_evals': self.MAX_EVALS, 'total_time':self.run_time, 'best_params':self.best_params}

        elif 'regression' in str(self.primitive_class):
            r2 = r2_score(test_target, prediction)
            explained_variance = explained_variance_score(test_target, prediction)
            scores_dict = {'optimization_technique': 'hyperopt', 'estimator': str(self.primitive_class), 'dataset': self.dataset_name, 'r2': r2, 'explained_variance_score': explained_variance, 'score': score, 'max_evals': self.MAX_EVALS, 'total_time':self.run_time, 'best_params':self.best_params}

        return scores_dict

    def save_trials(self, trials):
        path = self._save_to_folder('/hyperopt_{}_{}'.format(self.import_class, self.dataset_name), 'Trials.json')
        with open(path, 'w') as outfile:
            outfile.write(json.dumps(trials))