import numpy
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class BOHBWorker(Worker):
    def __init__(self, *args, sklearn_class, config,union_var,union_choice, data, target,**kwargs):
        super().__init__(*args, **kwargs)

        self.sklearn_class = sklearn_class
        self.config = config
        self.union_var = union_var
        self.union_choice = union_choice
        self.data = data
        self.target = target

    def compute(self, config, budget, **kwargs):
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/dataset fraction/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        cfg = {k: config[k] for k in config}

        # We translate None values:
        for item, key in cfg.items():
            if key == "None":
                cfg[item] = None
        print(cfg)
        cfg = self._translate_union_value(self.union_var, cfg)
        cfg = self._translate_union_value(self.union_choice, cfg)

        clf = self.sklearn_class(**cfg)
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=budget, random_state=0)
        scores = cross_val_score(clf, X_test, y_test, cv=5)
        loss = 1 - np.mean(scores)
        #
        # res = numpy.clip(config['x'] + numpy.random.randn() / budget, config['x'] / 2, 1.5 * config['x'])

        return ({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': {'loss':loss, 'params': cfg}  # can be used for any user-defined information - also mandatory
        })

    def get_configspace(self):
        return self.config

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