import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from hpbandster.core.worker import Worker

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
            'loss': loss,
            'info': {'loss':loss, 'params': cfg}  # used for user-defined info
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