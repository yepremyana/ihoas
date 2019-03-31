import numpy as np
import d3m.metadata.hyperparams as hyperparams
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score, precision_score, \
                            recall_score, f1_score
import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path
import csv
import json
import importlib
from timeit import default_timer as timer

class JPLGridSearch(object):
    def __init__(self, primitive_class, data, target, dataset_name='') -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.dataset_name = dataset_name
        self.parameters = {}
        self.best_params = None
        Path(current_dir + '/Results').mkdir(exist_ok=True, parents=True)
        self.current_dir = current_dir + '/Results'

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        self.import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, self.import_class)
        self.out_file = self.retrieve_path()

    def _enumeration_to_config_space(self, name, hp_value):
        params_config = hp_value.values

        return params_config

    def _constant_to_config_space(self, name, hp_value):
        default_hp = hp_value.get_default()
        params_config = [default_hp]
        return params_config

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
            params_config = np.linspace(lower, upper, num=(2*upper), endpoint=True)
        elif issubclass(structure_type, int):
            params_config = np.linspace(lower, upper, num=((upper-lower) + 1), dtype=np.int16, endpoint=True)

        return params_config

    def _union_to_config_space(self, name, hp_value):
        values_union = []
        for union_name, union_hp_value in hp_value.configuration.items():
            if isinstance(union_hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                child = self._bounded_to_config_space(union_name, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                child = self._enumeration_to_config_space(union_name, union_hp_value)
            elif isinstance(union_hp_value, (hyperparams.Constant)):
                child = self._constant_to_config_space(union_name, union_hp_value)
            values_union.extend(child)
        return values_union

    def _choice_to_config_space(self, name, hp_value):
        choice_combo = []
        for choice, hyperparameter in hp_value.choices.items():
            choice_dict = {}
            choice_dict[name] = [choice]
            for type, hp_info in hyperparameter.configuration.items():
                if type != 'choice':
                    if isinstance(hp_info, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                        values_union = self._bounded_to_config_space(type, hp_info)
                        choice_dict[type] =values_union
                    elif isinstance(hp_info, (hyperparams.Union)):
                        values_union = self._union_to_config_space(type, hp_info)
                        choice_dict[type] = values_union
            choice_combo.append(choice_dict)
        return choice_combo

    def _get_hp_search_space(self):
        hyperparameters = self.primitive_class.metadata.query()['primitive_code']['hyperparams']
        configuration = self.primitive_class.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        choice_used = False
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
                choice_used = True
                choice_combo = self._choice_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Constant)):
                params_config = self._constant_to_config_space(name, hp_value)
                self.parameters[name] = params_config

        param_grid = self.parameters
        if choice_used:
            param_grid = self._param_grid(choice_combo)

        return param_grid

    def _param_grid(self,  choice_combo):
        param_grid = []
        for item in choice_combo:
            item.update(self.parameters)
            param_grid.append(item)

        return param_grid

    def optimization(self):
        param_grid = self._get_hp_search_space()
        print(param_grid)
        clf = GridSearchCV(self.sklearn_class(), param_grid, cv=5)
        start = timer()
        clf.fit(self.data, self.target)
        print(sorted(clf.cv_results_['mean_test_score']))
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (timer() - start, len(clf.cv_results_['params'])))

        # File to save first results
        of_connection = open(self.out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['loss', 'params', 'iteration', 'train_time'])
        l = zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params'],np.arange(1, len(clf.cv_results_['params'])+1), clf.cv_results_['mean_fit_time'])
        for val in l:
            writer.writerow(val)
        of_connection.close()

        print('The parameters combination that would give best accuracy is : ')
        print(clf.best_params_)
        self.best_params = clf.best_params_
        print('The best accuracy achieved after parameter tuning via grid search is : ', clf.best_score_)
        print(clf.cv_results_)

    def _save_to_folder(self, path, savefig):
        Path(self.current_dir + path).mkdir(exist_ok=True, parents=True)
        return os.path.join(self.current_dir + path, savefig)

    def retrieve_path(self):
        return self._save_to_folder('/gridsearch_{}_{}'.format(self.import_class, self.dataset_name), 'Hyperparameter_Trials.csv')

    def validate(self, test_data, test_target):
        # test_score = self.sklearn_class(*self.best_params).fit(self.data, self.target).score(test_data, test_target)
        best_model = self.sklearn_class(**self.best_params)
        best_model.fit(self.data, self.target)
        prediction = best_model.predict(test_data)
        accuracy = accuracy_score(test_target, prediction)
        precision = precision_score(test_target, prediction, average=None)
        recall = recall_score(test_target, prediction, average=None)
        f1 = f1_score(test_target, prediction, average=None)
        r2 = r2_score(test_target, prediction)
        print('Accuracy Score : ', accuracy)
        print('Precision Score : ', precision)
        print('Recall Score : ', recall)
        print('F1 Score : ', f1)
        print('R2 score:', r2)

        return {'optimization_technique': 'gridsearch', 'dataset': self.dataset_name, 'accuracy_score': accuracy, 'precision_score': precision, 'recall_score': recall, 'f1_score': f1, 'prediction': prediction, 'r2_score': r2}

    # def save_trials(self, trials):
    #     path = self._save_to_folder('/gridsearch_{}_{}'.format(self.import_class, self.dataset_name), 'Trials.json')
    #     with open(path, 'w') as outfile:
    #         outfile.write(json.dumps(trials))

# # Logistic Regression (Grid Search) Confusion matrix
# confusion_matrix(y_test, y_pred_acc)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=0)
