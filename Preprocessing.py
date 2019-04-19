from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import pandas as pd

class Preprocessing(object):

    def __init__(self, data_id) -> None:
        self.data_id = data_id

    def load_data(self):
        dataset = fetch_openml(data_id = self.data_id, cache=False)
        return dataset

    def make_df(self, X_temp):
        col_names = self.dataset.feature_names
        X = pd.DataFrame(X_temp, columns = col_names)
        return X

    def _only_numerical_col(self):
        category_names = list(self.dataset.categories.keys())
        return list(set(category_names).symmetric_difference(set(self.dataset.feature_names)))

    def _only_categorical_col(self):
        return list(self.dataset.categories.keys())

    def label_encode(self, y_raw):
        y = LabelEncoder().fit_transform(y_raw)
        return y

    def standard_scaler(self, X):
        num_col = self._only_numerical_col()
        if not num_col:
            return X

        X[num_col] = StandardScaler().fit_transform(X[num_col])
        return X

    def simple_imputer_categorical(self, X):
        cat_col = self._only_categorical_col()
        if not cat_col:
            return X

        X[cat_col] = SimpleImputer(strategy = 'most_frequent').fit_transform(X[cat_col])
        return X

    def simple_imputer_numerical(self, X):
        num_col = self._only_numerical_col()
        if not num_col:
            return X

        X[num_col] = SimpleImputer(strategy='mean').fit_transform(X[num_col])
        return X

    def simple_preprocessing(self):
        self.dataset = self.load_data()
        y = self.dataset.target
        X_temp = self.dataset.data
        # Categorical features are encoded as ordinals from fetch_openml

        # impute missing values
        X = self.make_df(X_temp)
        X = self.simple_imputer_categorical(X)
        X = self.simple_imputer_numerical(X)
        # need to encode target columns
        if y.dtype == 'O':
            y = self.label_encode(y)

        # standard scaler
        X = self.standard_scaler(X)

        return X.to_numpy(), y
