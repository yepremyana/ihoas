import numpy as np
import d3m.metadata.hyperparams as hyperparams
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import importlib

class JPLGridSearch(object):
    def __init__(self, primitive_class, data, target) -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.parameters = {}

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, import_class)

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
        clf = GridSearchCV(self.sklearn_class(), param_grid, cv=5)
        clf.fit(self.data, self.target)
        print(sorted(clf.cv_results_['mean_test_score']))
        y_pred = clf.predict(self.data)
        print(y_pred)
        print('The parameters combination that would give best accuracy is : ')
        print(clf.best_params_)
        print('The best accuracy achieved after parameter tuning via grid search is : ', clf.best_score_)
