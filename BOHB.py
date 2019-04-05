import numpy as np
# from sklearn import datasets
import collections
import d3m.metadata.hyperparams as hyperparams
import ConfigSpace as CS
from hpbandster.optimizers import BOHB as BOHB
from BOHBWorker import BOHBWorker
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from sklearn.metrics import accuracy_score, r2_score, precision_score, \
                            recall_score, f1_score, explained_variance_score
from ConfigSpace.conditions import InCondition
import hpbandster.core.nameserver as hpns
import csv
import importlib
import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path
# from sklearn.ensemble.forest import RandomForestClassifier
# from d3m import index, utils

ITERATION = 0

class JPLBOHB(object):
    """
    Wrapped BOHB
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

    def optimization(self):
        cs = self._get_hp_search_space()
        print(self.cs)
        NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
        NS.start()

        w = BOHBWorker(sklearn_class=self.sklearn_class, config=self.cs, union_var=self.union_var, union_choice=self.union_choice,data=self.data,target=self.target,nameserver='127.0.0.1', run_id='example1')
        w.run(background=True)

        bohb = BOHB(configspace=w.get_configspace(),
                    run_id='example1', nameserver='127.0.0.1',
                    min_budget=0.1, max_budget=0.99
                    )
        res = bohb.run(n_iterations=self.MAX_EVALS)
        # bb_iterations = int(args.num_iterations * (1+(np.log(args.max_budget) - np.log(args.min_budget))/np.log(args.eta)))

        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        all_runs = res.get_all_runs()
        inc_runs = res.get_runs_by_id(incumbent)
        all_large_runs = res.get_all_runs(only_largest_budget=True)
        all_large_runs = list(filter(lambda r: r.budget== res.HB_config['max_budget'], all_large_runs))
        all_large_runs.sort(key=lambda r: r.loss)
        #filter by unique
        seen = set()
        unique = []
        for item in all_large_runs:
            if item.config_id[0] not in seen:
                unique.append(item)
                seen.add(item.config_id[0])

        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time'])
        of_connection.close()

        for item in unique:
            # params = id2config[item.config_id]['config']
            params = item.info['params']
            iteration_num = item.config_id[0]
            loss = item.loss
            time = item.time_stamps['finished'] - item.time_stamps['started']
            of_connection = open(self.out_file, 'a')
            writer = csv.writer(of_connection)
            writer.writerow([loss, params, iteration_num, time])
            of_connection.close()

        inc_run = inc_runs[-1]
        inc_loss = inc_run.loss
        best_params = list(filter(lambda r: r.budget== res.HB_config['max_budget'], res.get_runs_by_id(incumbent) ))
        self.best_params = best_params[0].info['params']
        self.run_time = all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']
        print('Best found configuration:', self.best_params)
        print('It achieved accuracies of %f (validation)'%(1-inc_loss))
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/81))
        print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))

    def _save_to_folder(self, path, savefig):
        Path(self.current_dir + path).mkdir(exist_ok=True, parents=True)
        return os.path.join(self.current_dir + path, savefig)

    def retrieve_path(self):
        return self._save_to_folder('/bohb_{}_{}'.format(self.import_class, self.dataset_name),
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
            scores_dict = {'optimization_technique': 'BOHB', 'estimator': str(self.primitive_class),
                           'dataset': self.dataset_name, 'r2': r2, 'explained_variance_score': explained_variance,
                           'score': score, 'max_evals': self.MAX_EVALS, 'total_time': self.run_time, 'best_params':cfg }

        return scores_dict

# import logging
# logging.basicConfig(level=logging.INFO)
#
# rng = np.random.RandomState(0)
# iris = datasets.load_iris()
# perm = rng.permutation(iris.target.size)
# # iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)
# primitive_class = index.get_primitive('d3m.primitives.classification.random_forest.SKlearn')
# smac = JPLBOHB(primitive_class, iris.data, iris.target)
# print(smac.optimization())