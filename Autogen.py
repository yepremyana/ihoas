import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from d3m import index
from SMAC import JPLSMAC
from GridSearch import JPLGridSearch
from HyperOpt import JPLHyperOpt
import logging
logging.basicConfig(level=logging.INFO)

rng = np.random.RandomState(0)
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data, iris.target = shuffle(iris.data, iris.target, random_state=rng)

def loop_through():
    all_primitives = index.search()
    for primitive in all_primitives:
        #if "SKlearn" in primitive:
        if "classification.random_forest.SKlearn" in primitive:
            primitive_obj = index.get_primitive(primitive)
            # smac = JPLSMAC(primitive_obj, iris.data, iris.target)
            # smac.optimization()
            # grid_search = JPLGridSearch(primitive_obj, iris.data, iris.target)
            # grid_search.optimization()
            hyperopt = JPLHyperOpt(primitive_obj, iris.data, iris.target)
            hyperopt.optimization()

if __name__ == "__main__":
    """
     This script tries to use the first given argument as the primitive, the
     second the dataset, the third is the benchmark.
     python 'd3m.primitives.classification.random_forest.SKlearn' iris_data.csv iris_target.csv benchmark.py
     
     - defaults can also be set
    """
    loop_through()