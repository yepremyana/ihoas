import pandas as pd
import argparse

from MetaData import MetaData
from datasets_id import classification_datasets_id, regression_datasets_id

def loop_through(args):
        if args.problem_type == 'classification':
            all_dataset_id = classification_datasets_id
        elif args.problem_type == 'regression':
            all_dataset_id = regression_datasets_id

        for data_id in all_dataset_id:
            md = MetaData(data_id)
            metafeature_dict = md.meta_features()
            metafeature_dict['data_id'] = data_id
            pd.DataFrame(metafeature_dict).to_csv('metafeatures_classification.csv', index=False)

if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_type", help="Problem type that is to be run", required=True)
    args = parser.parse_args()
    loop_through(args)