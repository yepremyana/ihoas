import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
import argparse
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from d3m import index

from HyperOpt import JPLHyperOpt
from SMAC import JPLSMAC
from BOHB import JPLBOHB
from Hyperband import HB
from Visual import Visual
from Preprocessing import Preprocessing
from datasets_id import classification_datasets_id, regression_datasets_id
from d3m_primitives_list import classification_primitives, regression_primitives

import logging
logging.basicConfig(level=logging.INFO)

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
        for i in range(3):
            X, y = shuffle(X_temp, y_temp)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            for primitive in all_primitives:
                print(primitive)
                primitive_obj = index.get_primitive(primitive)
                smac = JPLSMAC(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 600, rerun = i)
                smac.optimization()
                if i == 0:
                    all_visuals(smac)
                all_scores(smac, X_test, y_test, args.problem_type, rerun = i)
                bohb = JPLBOHB(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 600, rerun = i)
                bohb.optimization()
                if i == 0:
                    all_visuals(bohb)
                all_scores(bohb, X_test, y_test, args.problem_type, rerun = i)
                hb = HB(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 600, rerun = i)
                hb.optimization()
                if i == 0:
                    all_visuals(hb)
                all_scores(hb, X_test, y_test, args.problem_type, rerun = i)
                hyperopt = JPLHyperOpt(primitive_obj, X_train, y_train, dataset_name=data_id, max_evals = 600, rerun = i)
                hyperopt.optimization()
                if i == 0:
                    all_visuals(hyperopt)
                all_scores(hyperopt, X_test, y_test, args.problem_type, rerun = i)

def all_scores(algo, X_test, y_test, type_of_estimator, rerun):
    outfile = open('Results/{}_scores_{}.csv'.format(type_of_estimator, rerun), 'a')
    scores = algo.validate(X_test, y_test)
    writer = csv.DictWriter(outfile, scores.keys())
    if os.stat('Results/{}_scores_{}.csv'.format(type_of_estimator, rerun)).st_size == 0:
        writer.writeheader()
    writer.writerow(scores)
    outfile.close()

def all_visuals(algo):
    path = algo.retrieve_path()
    visual = Visual(path)
    visual.plot_all()

if __name__ == "__main__":
    """
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