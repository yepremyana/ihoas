import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from d3m import index
from SMAC import JPLSMAC
from BOHB import JPLBOHB
from datasets_id import classification_datasets_id, regression_datasets_id
from d3m_primitives_list import classification_primitives, regression_primitives
from Preprocessing import Preprocessing
from Visual import Visual
from HyperOpt import JPLHyperOpt
import logging
logging.basicConfig(level=logging.INFO)
import csv
rng = np.random.RandomState(0)
import os
import argparse

def loop_through(args):
    if args.problem_type == 'classification':
        all_dataset_id = classification_datasets_id
        all_primitives = classification_primitives
    elif args.problem_type == 'regression':
        all_dataset_id = regression_datasets_id
        all_primitives = regression_primitives

    for data_id in all_dataset_id:
        preprocess = Preprocessing(data_id = data_id)
        X_temp, y_temp = preprocess.simple_preprocessing()
        X, y = shuffle(X_temp, y_temp, random_state=rng)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        for primitive in all_primitives:
                print(primitive)
                primitive_obj = index.get_primitive(primitive)
                # smac = JPLSMAC(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 2)
                # smac.optimization()
                # all_visuals(smac)
                # all_scores(smac, X_test, y_test, args.problem_type)
                # bohb = JPLBOHB(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 2)
                # bohb.optimization()
                # all_visuals(smac)
                # all_scores(bohb, X_test, y_test, args.problem_type)
                hyperopt = JPLHyperOpt(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 4)
                hyperopt.optimization()
                # all_visuals(hyperopt)
                all_scores(hyperopt, X_test, y_test, args.problem_type)

def all_scores(algo, X_test, y_test, type_of_estimator):
    outfile = open('Results/{}_scores.csv'.format(type_of_estimator), 'a')
    scores = algo.validate(X_test, y_test)
    writer = csv.DictWriter(outfile, scores.keys())
    if os.stat('Results/{}_scores.csv'.format(type_of_estimator)).st_size == 0:
        writer.writeheader()
    writer.writerow(scores)
    outfile.close()
#add default to this
def all_visuals(algo):
    path = algo.retrieve_path()
    visual = Visual(path)
    visual.plot_all()

if __name__ == "__main__":
    """
     This script tries to use the first given argument as the primitive, the
     second the dataset, the third is the benchmark.
     python 'd3m.primitives.classification.random_forest.SKlearn' iris_data.csv iris_target.csv benchmark.py
     
     - defaults can also be set
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", help="Problem type that is to be run", required=True)
    args = parser.parse_args()
    loop_through(args)

# ToDO: Next steps
# add heat graphs 1) for iteration 2) for most similar
# compare the validation test scores across all and pick best one
# some sort choose the best param at 1 hr 2 hr 3hr and by evals
# add code for metadata datasets

# clean up code
# figure out how to run 5 random times
# where to save files
# make hyperband
# try and except in hyperband
# adding time since start
# add a time to stop and see how stopping is done in hpolib
# translate hyperopt and bohb
# fix hypdfopt in validation
# fix bohb in best_params