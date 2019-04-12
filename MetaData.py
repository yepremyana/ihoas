from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
# from scipy.stats import kurtosis, skew

class MetaData(object):

    def __init__(self, data_id) -> None:
        self.data_id = data_id

    def load_data(self):
        self.dataset = fetch_openml(data_id = self.data_id, cache=False)
        return

    def make_df(self):
        col_names = self.dataset.feature_names
        self.X = pd.DataFrame(self.dataset.data, columns = col_names)
        self.y = pd.DataFrame(self.dataset.target, columns = col_names)
        return

    def num_categories(self):
        return len(self.dataset.categories)

    def num_instance(self):
        rows, _ = self.X.shape
        return rows

    def num_features(self):
        _, columns = self.X.shape
        return columns

    def num_classes(self):
        _, columns = self.y.shape
        return columns

    def num_numeric_features(self):
        return self.num_features() - self.num_categories()

    def ratio_categorical_numerical(self):
        return self.num_categories()/self.num_numeric_features()

    def ratio_numerical_categorical(self):
        return self.num_numeric_features()/self.num_categories()

    def log_num_features(self):
        return np.log(self.num_features())

    def log_num_instances(self):
        return np.log(self.num_instance())

    def _only_numerical_col(self):
        #only the keys in the dictionary
        category_names = list(self.dataset.categories.keys())
        return list(set(category_names).difference(self.dataset.feature_names))

    def _kurtosis(self):
        numerical_c = self._only_numerical_col()
        numerical_df = self.X[numerical_c]
        return numerical_df.kurt(axis = 0, skipna = True)

    def kurtosis_mean(self):
        if not self._kurtosis():
            return np.nan

        return np.nanmean(self._kurtosis().values)

    def kurtosis_median(self):
        if not self._kurtosis():
            return np.nan

        return np.nanmedian(self._kurtosis().values)

    def kurtosis_std(self):
        if not self._kurtosis():
            return np.nan

        return np.nanstd(self._kurtosis().values)

    def kurtosis_min(self):
        if not self._kurtosis():
            return np.nan

        return np.min(self._kurtosis().values)

    def kurtosis_max(self):
        if not self._kurtosis():
            return np.nan

        return np.max(self._kurtosis().values)

    def _skew(self):
        numerical_c = self._only_numerical_col()
        numerical_df = self.X[numerical_c]
        return numerical_df.kurt(axis = 0, skipna = True)

    def meta_features(self):
        self.load_data()
        self.make_df()
        if self.dataset.default_target_attribute == 'class':
            self.num_classes()

        n_feature = self.num_features()

        self.ratio_categorical_numerical()
        self.ratio_numerical_categorical()
        #for does not have _ in the front
        #make a dataframe csv with all the features in it
        #count up the totals of each min, mean, max, value and time
        #scaling between 0 and 1

md = MetaData(3)
