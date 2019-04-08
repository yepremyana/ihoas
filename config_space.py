import collections
import d3m.metadata.hyperparams as hyperparams
import ConfigSpace as CS
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import InCondition

class Config_Space(object):
    """
    Wrapped Config_Space
    """

    def __init__(self, primitive_class) -> None:
        self.primitive_class = primitive_class
        self.cs = CS.ConfigurationSpace()
        self.union_var = []
        self.union_choice = []

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

    def get_union_var(self):
        return self.union_var

    def get_union_choice(self):
        return self.union_choice

    def get_hp_search_space(self):
        hyperparameters = self.primitive_class.metadata.query()['primitive_code']['hyperparams']
        configuration = self.primitive_class.metadata.query()['primitive_code']['class_type_arguments'][
            'Hyperparams'].configuration
        for name, description in hyperparameters.items():
            hp_value = configuration[name]
            if description['semantic_types'][0] == 'https://metadata.datadrivendiscovery.org/types/ControlParameter':
                continue
            elif isinstance(hp_value, (hyperparams.Enumeration, hyperparams.UniformBool)):
                self._enumeration_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Bounded, hyperparams.Uniform, hyperparams.UniformInt)):
                self._bounded_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Union)):
                self._union_to_config_space(name, hp_value)
                self.union_var.append(name)
            elif isinstance(hp_value, (hyperparams.Choice)):
                self._choice_to_config_space(name, hp_value)
            elif isinstance(hp_value, (hyperparams.Constant)):
                self._constant_to_config_space(name, hp_value)
        return self.cs