import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from Preprocessing import Preprocessing
# based on auto-sklearn meta-feature list

class MetaData(object):

    def __init__(self, data_id) -> None:
        self.data_id = data_id

    # def load_data(self):
    #     self.dataset = fetch_openml(data_id = self.data_id, cache=False)
    #     return

    def make_df(self, X_temp, y_temp):
        col_names = self.dataset.feature_names
        self.X = pd.DataFrame(X_temp, columns = col_names)
        self.y = pd.Series(y_temp)
        return

    def num_categories(self):
        return len(self.dataset.categories)

    def num_instance(self):
        # number of data-points
        rows, _ = self.X.shape
        return rows

    def num_features(self):
        _, columns = self.X.shape
        return columns

    def num_classes(self):
        num_classes = len(self.y.unique())
        return num_classes

    def num_numeric_features(self):
        return self.num_features() - self.num_categories()

    def ratio_categorical_numerical(self):
        # ratio of number of categorical features to numeric features
        if self.num_numeric_features() == 0:
            return 1
        elif self.num_categories() == 0:
            return 0

        return self.num_categories()/self.num_numeric_features()

    def ratio_numerical_categorical(self):
        if self.num_numeric_features() == 0:
            return 0
        elif self.num_categories() == 0:
            return 1

        return self.num_numeric_features()/self.num_categories()

    def log_num_features(self):
        return np.log(self.num_features())

    def log_num_instances(self):
        return np.log(self.num_instance())

    # def _only_categorical_col(self):
    #     return list(self.dataset.categories.keys())

    def dataset_ratio(self):
        #the ratio of num_features to the number of data points
        return self.num_features()/self.num_instance()

    def inverse_dataset_ratio(self):
        return self.num_instance()/self.num_features()

    def log_dataset_ratio(self):
        return np.log(self.dataset_ratio())

    def log_inverse_dataset_ratio(self):
        return np.log(self.inverse_dataset_ratio())

    def _class_prob(self):
        # ratios of data-points in each class (the proportion of instances in data-set that have class C)
        return self.y.value_counts()/self.num_instance()

    def class_cross_entropy(self):
        #l log based 2
        # calculate a separate loss for each class label per observation and sum the result. entropy of class labels
        entropy = -1 * self._class_prob() * np.log2(self._class_prob())
        return entropy.sum()

    def class_prob_mean(self):
        return self._class_prob().mean()

    def class_prob_median(self):
        return self._class_prob().median()

    def class_prob_std(self):
        return self._class_prob().std()

    def class_prob_min(self):
        return self._class_prob().min()

    def class_prob_max(self):
        return self._class_prob().max()

    def _symbol(self):
        # number of unique symbols for each categorical feature
        symbol_count = []
        for key, value in self.dataset.categories.items():
            symbol_count.append(len(value))

        return symbol_count

    def symbol_mean(self):
        if not self._symbol():
            return np.nan

        return np.nanmean(self._symbol())

    def symbol_sum(self):
        if not self._symbol():
            return np.nan

        return np.sum(self._symbol())

    def symbol_std(self):
        if not self._symbol():
            return np.nan

        return np.nanstd(self._symbol())

    def symbol_min(self):
        if not self._symbol():
            return np.nan

        return np.min(self._symbol())

    def symbol_max(self):
        if not self._symbol():
            return np.nan

        return np.max(self._symbol())

    def _only_numerical_col(self):
        #only the keys in the dictionary
        category_names = list(self.dataset.categories.keys())
        return list(set(category_names).symmetric_difference(set(self.dataset.feature_names)))

    def _kurtosis(self):
        numerical_c = self._only_numerical_col()
        numerical_df = self.X[numerical_c]
        return numerical_df.kurt(axis = 0, skipna = True)

    def kurtosis_mean(self):
        if self._kurtosis().empty:
            return np.nan

        return np.nanmean(self._kurtosis().values)

    def kurtosis_median(self):
        if self._kurtosis().empty:
            return np.nan

        return np.nanmedian(self._kurtosis().values)

    def kurtosis_std(self):
        if self._kurtosis().empty:
            return np.nan

        return np.nanstd(self._kurtosis().values)

    def kurtosis_min(self):
        if self._kurtosis().empty:
            return np.nan

        return np.min(self._kurtosis().values)

    def kurtosis_max(self):
        if self._kurtosis().empty:
            return np.nan

        return np.max(self._kurtosis().values)

    def _skew(self):
        numerical_c = self._only_numerical_col()
        numerical_df = self.X[numerical_c]
        return numerical_df.skew(axis = 0, skipna = True)

    def skew_mean(self):
        if self._skew().empty:
            return np.nan

        return np.nanmean(self._skew().values)

    def skew_median(self):
        if self._skew().empty:
            return np.nan

        return np.nanmedian(self._skew().values)

    def skew_std(self):
        if self._skew().empty:
            return np.nan

        return np.nanstd(self._skew().values)

    def skew_min(self):
        if self._skew().empty:
            return np.nan

        return np.min(self._skew().values)

    def skew_max(self):
        if self._skew().empty:
            return np.nan

        return np.max(self._skew().values)

    def _get_pca(self):
        # we want to retain 95% of the variance with the least amount of dimensions
        # fraction of components that account for 95% of the variance
        # X_std = StandardScaler().fit_transform(self.X)
        clf = PCA(copy=True).fit(self.X)
        return clf

    def pca_fraction_95(self):
        variance = self._get_pca().explained_variance_ratio_

        sum_var = 0
        i = 0
        while sum_var<0.95 and i<len(variance):
            sum_var += variance[i]
            i += 1

        return i/len(variance)

    def pca_kurtosis(self):
        # kurtosis of dimensionality reduction along the first principle component
        # X_reduced = self._get_pca().transform(X_std)
        # X_first_pc = pd.DataFrame(X_reduced[0])
        clf= self._get_pca()
        clf.components_ = clf.components_[:1]
        X_reduced = clf.transform(self.X)
        X_first_pc = pd.DataFrame(X_reduced)
        return X_first_pc.kurt(axis=0, skipna=True)

    def pca_skew(self):
        # skew of dimensionality reduction along the first principle component
        clf = self._get_pca()
        clf.components_ = clf.components_[:1]
        X_reduced = clf.transform(self.X)
        X_first_pc = pd.DataFrame(X_reduced)
        return X_first_pc.skew(axis=0, skipna=True)

    def _raw_data(self):
        return pd.DataFrame(self.dataset.data)

    def num_instances_missing_values(self):
        instances_missing = self._raw_data().isna().sum(axis=1)
        instances_missing = instances_missing[instances_missing != 0]
        return len(instances_missing)

    def num_features_missing_values(self):
        features_missing = self._raw_data().isna().sum(axis = 0)
        features_missing = features_missing[features_missing != 0]
        return len(features_missing)

    def percentage_instances_missing(self):
        return self.num_features_missing_values()/self.num_instance()

    def percentage_features_missing(self):
        return self.num_features_missing_values() / self.num_features()

    def num_missing(self):
        return sum(self._raw_data().isna().sum(axis = 1))

    def percentage_missing(self):
        return self.num_missing()/(self.num_instance()*self.num_features())

    def landmark_1nn(self):
        try:
            clf = KNeighborsClassifier(n_neighbors=1)
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def landmark_decision_node_learner(self):
        try:
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=42,
                                         min_samples_split=2, min_samples_leaf=1,  max_features=None)
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def landmark_decision_tree(self):
        try:
            clf = DecisionTreeClassifier(random_state = 42)
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def landmark_lda(self):
        try:
            clf = LinearDiscriminantAnalysis()
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def landmark_naive_bayes(self):
        try:
            clf = GaussianNB()
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def landmark_random_node_learner(self):
        try:
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=42,
                                         min_samples_split=2, min_samples_leaf=1,  max_features=1)
            scores = cross_val_score(clf, self.X, self.y, cv=10)
            return np.mean(scores)
        except:
            return np.nan

    def meta_features(self):
        preprocess = Preprocessing(data_id = self.data_id)
        X_temp, y_temp = preprocess.simple_preprocessing()
        self.dataset = preprocess.load_data()
        # self.load_data()
        self.make_df(X_temp, y_temp)
        # metafeature_dict = {}
        n_classes = np.nan
        if self.dataset.details['default_target_attribute'] == 'class':
            n_classes = self.num_classes()
            # metafeature_dict['num_classes'] = n_classes

        metafeature_dict = {'num_features': self.num_features(),
                            'num_instance': self.num_instance(),
                            'num_classes': n_classes,
                            'num_categories': self.num_categories(),
                            'num_numeric_features': self.num_numeric_features(),
                            'ratio_categorical_numerical': self.ratio_categorical_numerical(),
                            'ratio_numerical_categorical': self.ratio_numerical_categorical(),
                            'log_num_features': self.log_num_features(),
                            'log_num_instances': self.log_num_instances(),
                            'dataset_ratio': self.dataset_ratio(),
                            'inverse_dataset_ratio': self.inverse_dataset_ratio(),
                            'log_dataset_ratio': self.log_dataset_ratio(),
                            'log_inverse_dataset_ratio': self.log_inverse_dataset_ratio(),
                            'class_cross_entropy': self.class_cross_entropy(),
                            'class_prob_mean': self.class_prob_mean(),
                            'class_prob_median': self.class_prob_median(),
                            'class_prob_std': self.class_prob_std(),
                            'class_prob_min': self.class_prob_min(),
                            'class_prob_max': self.class_prob_max(),
                            'symbol_mean': self.symbol_mean(),
                            'symbol_sum': self.symbol_sum(),
                            'symbol_std': self.symbol_std(),
                            'symbol_min': self.symbol_min(),
                            'symbol_max': self.symbol_max(),
                            'kurtosis_mean': self.kurtosis_mean(),
                            'kurtosis_median': self.kurtosis_median(),
                            'kurtosis_std': self.kurtosis_std(),
                            'kurtosis_min': self.kurtosis_min(),
                            'kurtosis_max': self.kurtosis_max(),
                            'skew_mean': self.skew_mean(),
                            'skew_median': self.skew_median(),
                            'skew_std': self.skew_std(),
                            'skew_min': self.skew_min(),
                            'skew_max': self.skew_max(),
                            'pca_fraction_95': self.pca_fraction_95(),
                            'pca_kurtosis': self.pca_kurtosis(),
                            'pca_skew': self.pca_skew(),
                            'num_instances_missing_values': self.num_instances_missing_values(),
                            'num_features_missing_values': self.num_features_missing_values(),
                            'percentage_instances_missing': self.percentage_instances_missing(),
                            'percentage_features_missing': self.percentage_features_missing(),
                            'num_missing': self.num_missing(),
                            'percentage_missing': self.percentage_missing(),
                            'landmark_1nn': self.landmark_1nn(),
                            'landmark_decision_node_learner': self.landmark_decision_node_learner(),
                            'landmark_decision_tree': self.landmark_decision_tree(),
                            'landmark_lda': self.landmark_lda(),
                            'landmark_naive_bayes': self.landmark_naive_bayes(),
                            'landmark_random_node_learner': self.landmark_random_node_learner()
        }

        return metafeature_dict

        #count up the totals of each min, mean, max, value and time