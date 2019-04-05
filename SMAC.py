import numpy as np
import collections
import d3m.metadata.hyperparams as hyperparams
import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from sklearn.metrics import accuracy_score, r2_score, precision_score, \
                            recall_score, f1_score, explained_variance_score
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer
import csv
import importlib
import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path

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
        self.cs = CS.ConfigurationSpace()
        self.union_var = []
        self.union_choice = []
        self.MAX_EVALS = max_evals
        self.best_params = None
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
        default_hp = hp_value.get_default()
        values = hp_value.values
        params_config = CategoricalHyperparameter(name=name,
                                                  choices=values,
                                                  default_value=default_hp)
        self.cs.add_hyperparameter(params_config)
        return params_config

    def _constant_to_config_space(self, name, hp_value):
        default_hp = hp_value.get_default()
        if default_hp == None:
            default_hp = str(None)
        params_config = Constant(name=name, value=default_hp)
        self.cs.add_hyperparameter(params_config)
        return params_config

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
        # when hyperparameter class is float use UniformFloatHyperparameter
        if issubclass(structure_type, float):
            params_config = UniformFloatHyperparameter(name=name,
                                                       lower=lower, upper=upper,
                                                       default_value=default_hp)

        # when hyperparameter class is int use UniformIntegerHyperparameter
        elif issubclass(structure_type, int):
            params_config = UniformIntegerHyperparameter(name=name,
                                                         lower=lower, upper=upper,
                                                         default_value=default_hp)

        self.cs.add_hyperparameter(params_config)
        return params_config

    def _union_to_config_space(self, name, hp_value):
        union_child = []
        union_config = []
        for union_name, union_hp_value in hp_value.configuration.items():
            unique_union_name = "{}_{}".format(name, union_name)
            if isinstance(union_hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                child = self._bounded_to_config_space(unique_union_name, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                child = self._enumeration_to_config_space(unique_union_name, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Constant)):
                child = self._constant_to_config_space(unique_union_name, union_hp_value)
            union_child.append(unique_union_name)
            union_config.append(child)

        # params_config = CategoricalHyperparameter(name=name, choices=union_child, default_value=hp_value.get_default())
        params_config = CategoricalHyperparameter(name=name, choices=union_child)
        self.cs.add_hyperparameter(params_config)
        [self.cs.add_condition(InCondition(child=item, parent=params_config, values=[item.name])) for item in union_config]
        return params_config

    def _choice_to_config_space(self, name, hp_value):
        default_hp = (hp_value.get_default().configuration['choice'].get_default())
        values = list(hp_value.choices.keys())
        parent_config = CategoricalHyperparameter(name=name,
                                                  choices=values,
                                                  default_value=default_hp)
        self.cs.add_hyperparameter(parent_config)

        choice_dict = collections.defaultdict(list)
        conditions = []
        for choice, hyperparameter in hp_value.choices.items():
            for type, hp_info in hyperparameter.configuration.items():
                if type != 'choice':
                    if type in self.cs.get_hyperparameter_names():
                        # remove the condition from the cs space
                        for item in conditions:
                            if item.child.name != type:
                                continue
                            else:
                                choice_dict[type].append(item.value)
                                conditions.remove(item)
                        choice_dict[type].append(choice)
                        continue
                    elif isinstance(hp_info, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                        type_config = self._bounded_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                    elif isinstance(hp_info, (hyperparams.Constant)):
                        type_config = self._constant_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                    elif isinstance(hp_info, (hyperparams.Enumeration)):
                        type_config = self._enumeration_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                    elif isinstance(hp_info, (hyperparams.Union)):
                        type_config = self._union_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                        self.union_choice.append(type)
                    conditions.append(child_choice)
        self.cs.add_conditions(conditions)

        for key, value in choice_dict.items():
            arg_list = []
            cs = self.cs.get_hyperparameters()
            for idx in range(len(cs)):
                if cs[idx].name == key:
                    child = cs[idx]
                else:
                    continue
            [arg_list.append(CS.EqualsCondition(child, parent_config, key)) for key in value]
            or_conj = CS.OrConjunction(*arg_list)
            self.cs.add_condition(or_conj)

        return parent_config

    def _get_hp_search_space(self):
        hyperparameters = self.primitive_class.metadata.query()['primitive_code']['hyperparams']
        configuration = self.primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        for name, description in hyperparameters.items():
            hp_value = configuration[name]
            if description['semantic_types'][0] == 'https://metadata.datadrivendiscovery.org/types/ControlParameter':
                continue
            elif isinstance(hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                params_config = self._enumeration_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                params_config = self._bounded_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Union)):
                params_config = self._union_to_config_space(name, hp_value)
                self.union_var.append(name)
            elif isinstance(hp_value, (hyperparams.Choice)):
                params_config = self._choice_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Constant)):
                params_config = self._constant_to_config_space(name, hp_value)
        return params_config

    def primitive_from_cfg(self, cfg):
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
        cs = self._get_hp_search_space()
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

        # def_value = self.primitive_from_cfg(self.cs.get_default_configuration())
        # print("Default Value: %.2f" % (def_value))

        print("Optimizing! Depending on your machine, this might take a few minutes.")
        start = timer()
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=self.primitive_from_cfg)
        incumbent = smac.optimize()
        self.best_params = incumbent
        run_time = timer() - start
        self.run_time = run_time
        # inc_value = self.primitive_from_cfg(incumbent)
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

    def validate(self, test_data, test_target, pos_label, average):
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
            f1 = f1_score(test_target, prediction, average=average, pos_label=pos_label)
            accuracy = accuracy_score(test_target, prediction)
            precision = precision_score(test_target, prediction, average=average, pos_label=pos_label)
            recall = recall_score(test_target, prediction, average=average, pos_label=pos_label)
            # confusion matrix
            scores_dict = {'optimization_technique': 'smac', 'estimator': str(self.primitive_class), 'dataset': self.dataset_name,
                           'accuracy_score': accuracy, 'precision_score': precision, 'recall_score': recall,
                           'f1_score': f1, 'prediction': prediction, 'score': score, 'max_evals': self.MAX_EVALS, 'total_time': self.run_time, 'best_params':cfg}

        elif 'regression' in str(self.primitive_class):
            r2 = r2_score(test_target, prediction)
            explained_variance = explained_variance_score(test_target, prediction)
            scores_dict = {'optimization_technique': 'smac', 'estimator': str(self.primitive_class),
                           'dataset': self.dataset_name, 'r2': r2, 'explained_variance_score': explained_variance,
                           'score': score, 'max_evals': self.MAX_EVALS, 'total_time': self.run_time, 'best_params':cfg }

        return scores_dict

            # need to make a utils file with basic things list lower, upper, save to folder, retrieve path
# need to now work with the validation code that they have and whatever else
# need to check the other functions SMAC has
# fix overlay files, d3m issue, fix hyperparameter type parameters
# need to fix the default in union, it is wrong bc i am not setting the choices default but configuration wont give it to me, self.default_hyperparameter configuration[default]
# understand why global iteration wont work