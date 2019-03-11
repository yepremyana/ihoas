import numpy as np
import d3m.metadata.hyperparams as hyperparams
import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.model_selection import cross_val_score
import importlib

class JPLSMAC(object):
    """
    Wrapped SMAC
    """

    def __init__(self, primitive_class, data, target) -> None:
        self.primitive_class = primitive_class
        self.data = data
        self.target = target
        self.cs = CS.ConfigurationSpace()
        self.union_var = []
        self.union_choice = []

        primitive_json = primitive_class.metadata.query().get('name')
        import_module = ".".join(primitive_json.split(".")[:-1])
        sklearn_module = importlib.import_module(import_module)
        import_class = primitive_json.split(".")[-1]
        self.sklearn_class = getattr(sklearn_module, import_class)

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
                upper = 2 * (default_hp)
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

        for choice, hyperparameter in hp_value.choices.items():
            for type, hp_info in hyperparameter.configuration.items():
                if type != 'choice':
                    if isinstance(hp_info, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                        type_config = self._bounded_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                    elif isinstance(hp_info, (hyperparams.Union)):
                        type_config = self._union_to_config_space(type, hp_info)
                        child_choice = CS.EqualsCondition(type_config, parent_config, choice)
                        self.union_choice.append(type)
                    self.cs.add_condition(child_choice)
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
        cfg = {k: cfg[k] for k in cfg}

        # We translate None values:
        for item, key in cfg.items():
            if key == "None":
                cfg[item] = None
        print(cfg)
        cfg = self._translate_union_value(self.union_var, cfg)
        cfg = self._translate_union_value(self.union_choice, cfg)

        clf = self.sklearn_class(**cfg, random_state=42)

        scores = cross_val_score(clf, self.data, self.target, cv=5)
        return 1 - np.mean(scores)  # Minimize!

    def _translate_union_value(self, union_list, cfg):
        # We translate Union values:
        for item in union_list:
            value = cfg[item]
            cfg[item] = cfg[value]
            cfg.pop(value, None)  # Remove extra union choices from config
        return cfg

    def optimization(self):
        cs = self._get_hp_search_space()
        print(self.cs)
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": 200,  # maximum function evaluations
                             "cs": self.cs,  # configuration space
                             "deterministic": "false",
                             "memory_limit": 100,
                             })

        def_value = self.primitive_from_cfg(self.cs.get_default_configuration())
        print("Default Value: %.2f" % (def_value))

        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=self.primitive_from_cfg)
        incumbent = smac.optimize()

        inc_value = self.primitive_from_cfg(incumbent)
        print("Optimized Value: %.2f" % (inc_value))

        return inc_value

# need to make a utils file with basic things list lower, upper
# need to now work with the validation code that they have and whatever else
# need to check the other functions SMAC has
# fix overlay files, d3m issue, fix hyperparameter type parameters
# need to fix the default in union, it is wrong bc i am not setting the choices default but configuration wont give it to me, self.default_hyperparameter configuration[default]