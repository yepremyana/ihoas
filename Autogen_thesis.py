import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from d3m import index
from SMAC import JPLSMAC
from GridSearch import JPLGridSearch
from Visual import Visual
from HyperOpt import JPLHyperOpt
import logging
logging.basicConfig(level=logging.INFO)

rng = np.random.RandomState(0)

#['fertility', version=1, cache=False],['blogger', version=1, cache=False,]['nursery', version=3, cache=False]['parkinsons', version=1, cache=False]
# all_datasets = [['fertility', 1], ['blogger', 1], ['nursery', 3], ['parkinsons', 1]]
all_datasets = [['fertility', 1]]

all_primitives = ['d3m.primitives.classification.random_forest.SKlearn']
def loop_through():
    for dataset_arg in all_datasets:
        imp = SimpleImputer()
        dataset = fetch_openml(*dataset_arg, cache=False)
        X_temp, y = shuffle(dataset.data, dataset.target, random_state=rng)
        X = imp.fit_transform(X_temp)
        for primitive in all_primitives:
                primitive_obj = index.get_primitive(primitive)
                smac = JPLSMAC(primitive_obj, X, y, dataset_name=dataset_arg[0], max_evals = 10)
                smac.optimization()
                # results_path = smac.retrieve_path()
                # all_visuals(results_path)
                # grid_search = JPLGridSearch(primitive_obj, X, y, dataset_name=dataset_arg[0])
                # grid_search.optimization()
                # grid_results_path = grid_search.retrieve_path()
                # all_visuals(grid_results_path)
                # hyperopt = JPLHyperOpt(primitive_obj, X, y, dataset_name=dataset_arg[0], max_evals = 10)
                # hyperopt.optimization()
                # results_path = hyperopt.retrieve_path()
                # all_visuals(results_path)

def all_visuals(path):
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

# Next steps
# add heat graphs 1) for iteration 2) for most similar
# compare the validation test scores across all and pick best one
# some sort choose the best param at 1 hr 2 hr 3hr
# add a default validation score
# add code for metadata datasets
# pick the datasets and the estimators that will be used for thesis
# clean up code in general