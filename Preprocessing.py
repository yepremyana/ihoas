from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class Preprocessing(object):

    def __init__(self, data_id) -> None:
        self.data_id = data_id

    def load_data(self):
        dataset = fetch_openml(data_id = self.data_id, cache=False)
        return dataset

    def label_encode(self, y_raw):
        y = LabelEncoder().fit_transform(y_raw)
        return y

    def standard_scaler(self, X):
        X = StandardScaler().fit_transform(X)
        return X

    def simple_imputer(self, X):
        X = SimpleImputer(strategy = 'most_frequent').fit_transform(X)
        return X

    def simple_preprocessing(self):
        dataset = self.load_data()
        y = dataset.target
        X = dataset.data
        # Categorical features are encoded as ordinals from fetch_openml

        # impute missing values
        X = self.simple_imputer(X)
        # need to encode target columns
        if y.dtype == 'O':
            y = self.label_encode(y)

        # standard scaler
        X = self.standard_scaler(X)

        return X, y
