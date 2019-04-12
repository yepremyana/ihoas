import csv
import pandas as pd
import importlib
from pathlib import Path

from sklearn.metrics import accuracy_score, r2_score, precision_score, mean_squared_error,\
                            recall_score, f1_score, explained_variance_score, roc_auc_score
from sklearn.utils.multiclass import type_of_target

from hpbandster.optimizers import HyperBand
import hpbandster.core.nameserver as hpns
from BOHBWorker import BOHBWorker

import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

from config_space import Config_Space

ITERATION = 0

class HB(object):
    """
    Wrapped HB
    """

    def __init__(self, primitive_class, data, target, dataset_name='',max_evals=50, rerun='') -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.dataset_name = dataset_name
        self.MAX_EVALS = max_evals
        self.rerun = rerun
        Path(current_dir + '/Results').mkdir(exist_ok=True, parents=True)
        self.current_dir = current_dir + '/Results'

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        self.import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, self.import_class)
        self.out_file = self.retrieve_path()
        self.run_id = 'hb_{}_{}'.format(self.import_class, self.dataset_name)

    def optimization(self):
        config_space = Config_Space(self.primitive_class)
        self.cs = config_space.get_hp_search_space()
        self.union_var = config_space.get_union_var()
        self.union_choice = config_space.get_union_choice()
        print(self.cs)

        NS = hpns.NameServer(run_id=self.run_id, host='127.0.0.1', port=None)
        NS.start()

        w = BOHBWorker(sklearn_class=self.sklearn_class, config=self.cs, union_var=self.union_var,
                       union_choice=self.union_choice,data=self.data,target=self.target,nameserver='127.0.0.1',
                       run_id=self.run_id)
        w.run(background=True)

        hb = HyperBand(configspace=w.get_configspace(),
                    run_id=self.run_id, nameserver='127.0.0.1',
                    min_budget=0.1, max_budget=0.99
                    )
        res = hb.run(n_iterations=self.MAX_EVALS)
        # bb_iterations = int(args.num_iterations * (1+(np.log(args.max_budget) - np.log(args.min_budget))/np.log(args.eta)))

        hb.shutdown(shutdown_workers=True)
        NS.shutdown()
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        all_runs = res.get_all_runs()
        inc_runs = res.get_runs_by_id(incumbent)
        all_large_runs = res.get_all_runs(only_largest_budget=True)
        all_large_runs = list(filter(lambda r: r.budget== res.HB_config['max_budget'], all_large_runs))
        all_large_runs_clean = [run for run in all_large_runs if run.loss]
        all_large_runs_clean.sort(key=lambda r: r.loss)
        #filter by unique
        seen = set()
        unique = []
        for item in all_large_runs_clean:
            if item.config_id[0] not in seen:
                unique.append(item)
                seen.add(item.config_id[0])

        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time', 'cum_train_time'])
        of_connection.close()

        for item in unique:
            # params = id2config[item.config_id]['config']
            params = item.info['params']
            iteration_num = item.config_id[0]
            loss = item.loss
            time = item.time_stamps['finished'] - item.time_stamps['started']
            cum_time = item.time_stamps['finished'] - all_runs[0].time_stamps['started']
            of_connection = open(self.out_file, 'a')
            writer = csv.writer(of_connection)
            writer.writerow([loss, params, iteration_num, time, cum_time])
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
        return self._save_to_folder('/hb_{}_{}'.format(self.import_class, self.dataset_name),
                                    'Hyperparameter_Trials_{}.csv'.format(self.rerun))

    def _classification_scoring(self, test_target, prediction, average_type=None, positive_label=1):
        accuracy = accuracy_score(test_target, prediction)
        f1 = f1_score(test_target, prediction, average=average_type, pos_label=positive_label)
        precision = precision_score(test_target, prediction, average=average_type, pos_label=positive_label)
        recall = recall_score(test_target, prediction, average=average_type, pos_label=positive_label)

        return {'optimization_technique': 'hb', 'estimator': str(self.primitive_class),
                'dataset': self.dataset_name, 'accuracy_score': accuracy, 'precision_score': precision,
                'recall_score': recall, 'f1_score': f1,
                'max_evals': self.MAX_EVALS, 'total_time': self.run_time,
                'best_params': self.best_params}

    def validate(self, test_data, test_target):

        best_model = self.sklearn_class(**self.best_params)
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
            scores_dict = {'optimization_technique': 'hb', 'estimator': str(self.primitive_class),
                           'dataset': self.dataset_name, 'r2': r2, 'explained_variance_score': explained_variance,
                           'mean_squared_error': mse, 'max_evals': self.MAX_EVALS,
                           'total_time':self.run_time,'best_params':self.best_params,'score': score}

        return scores_dict