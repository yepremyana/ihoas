import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from d3m import index
from SMAC import JPLSMAC
from GridSearch import JPLGridSearch
from Visual import Visual
from HyperOpt import JPLHyperOpt
import logging
logging.basicConfig(level=logging.INFO)
import csv
rng = np.random.RandomState(0)
import os

#['fertility', version=1, cache=False],['blogger', version=1, cache=False,]['nursery', version=3, cache=False]['parkinsons', version=1, cache=False]
# all_datasets = [['fertility', 1], ['blogger', 1], ['nursery', 3], ['parkinsons', 1]]
all_datasets = [[['fertility', 1], 'binary', '1', 'classification'] ]

all_primitives = ['d3m.primitives.classification.random_forest.SKlearn', 'd3m.primitives.classification.svc.SKlearn']
def loop_through():
    for dataset_arg in all_datasets:
        imp = SimpleImputer()
        dataset = fetch_openml(*dataset_arg[0], cache=False)
        X_temp, y = shuffle(dataset.data, dataset.target, random_state=rng)
        X = imp.fit_transform(X_temp)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        for primitive in all_primitives:
                primitive_obj = index.get_primitive(primitive)
                smac = JPLSMAC(primitive_obj, X_train, y_train, dataset_name=dataset_arg[0][0], max_evals = 20)
                smac.optimization()
                all_visuals(smac)
                all_scores(smac, X_test, y_test, dataset_arg[3], dataset_arg[2], dataset_arg[1])
                # grid_search = JPLGridSearch(primitive_obj, X_train, y_train, dataset_name=dataset_arg[0])
                # grid_search.optimization()
                # all_visuals(grid_search)
                # all_scores(grid_search, X_test, y_test)
                hyperopt = JPLHyperOpt(primitive_obj, X, y, dataset_name=dataset_arg[0][0], max_evals = 20)
                hyperopt.optimization()
                all_visuals(hyperopt)
                all_scores(hyperopt, X_test, y_test, dataset_arg[3], dataset_arg[2], dataset_arg[1])

def all_scores(algo, X_test, y_test, type_of_estimator, pos_label, average):
    outfile = open('Results/{}_scores.csv'.format(type_of_estimator), 'a')
    scores = algo.validate(X_test, y_test, pos_label, average)
    writer = csv.DictWriter(outfile, scores.keys())
    if os.stat('Results/{}_scores.csv'.format(type_of_estimator)).st_size == 0:
        writer.writeheader()
    writer.writerow(scores)
    outfile.close()
#add default to this
def all_visuals(algo):
    path = algo.retrieve_path()
    visual = Visual(path)
    visual.density_parameters()
    visual.numerical_evolution()
    visual.sort_best_scores()
    visual.categorical_bar_graph()
    visual.categorical_evolution()
    visual.density_loss()

if __name__ == "__main__":
    """
     This script tries to use the first given argument as the primitive, the
     second the dataset, the third is the benchmark.
     python 'd3m.primitives.classification.random_forest.SKlearn' iris_data.csv iris_target.csv benchmark.py
     
     - defaults can also be set
    """
    loop_through()

# ToDO: Next steps
# add heat graphs 1) for iteration 2) for most similar
# compare the validation test scores across all and pick best one
# some sort choose the best param at 1 hr 2 hr 3hr and by evals
# add a default validation score
# add code for metadata datasets
# pick the datasets that will be used for thesis
# clean up code in general
# add spearmint and another hyper techinque and randomsearch