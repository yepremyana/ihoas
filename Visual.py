import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path
import pandas as pd
import numpy as np
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
import ast
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

class Visual(object):

    def __init__(self, file) -> None:
        self.file = file
        self.results = pd.read_csv(self.file)
        self.current_dir = os.path.join(current_dir, os.path.splitext(file)[0])

    def sort_best_scores(self):
        results = self.results

        # Sort with best scores on top and reset index for slicing
        results.sort_values('loss', ascending = True, inplace = True)
        results.reset_index(inplace = True, drop = True)
        print(results)

        return

    def _dataframe_parameters(self):
        results_parameters = []
        [results_parameters.append(ast.literal_eval(item)) for item in list(self.results['params'])]

        return pd.DataFrame(results_parameters)

    def density_parameters(self, params=None):
        df_parameters = self._dataframe_parameters()
        if not params:
            params = list(df_parameters)

        for item in params:
            df_parameters[item] = pd.to_numeric(df_parameters[item], errors='coerce')
            item_series = df_parameters[item].dropna()
            #remove the bool ans as well

            plt.figure(figsize=(20, 8))
            plt.rcParams['font.size'] = 18

            if not item_series.empty:
                sns.kdeplot(item_series, label=item, linewidth=2).set_title('{} distribution'.format(item))
                plt.xlabel(item)
                plt.ylabel('Density')
                path = self._save_to_folder('/density_parameters','{}_distribution.pdf'.format(item))
                plt.savefig(path)

            else:
                continue

    def _save_to_folder(self, path, savefig):
        Path(self.current_dir + path).mkdir(exist_ok=True, parents=True)
        return os.path.join(self.current_dir + path, savefig)

    def categorical_bar_graph(self, params=None):
        df_parameters = self._dataframe_parameters()
        if not params:
            params = list(df_parameters)

        for item in params:
            if is_float_dtype(df_parameters[item]) or is_integer_dtype(df_parameters[item]):
                continue
            else:
                for i in range(len(df_parameters[item].index)):
                    if type(df_parameters[item][i]) == str or type(df_parameters[item][i]) == bool:
                        continue
                    else:
                        df_parameters[item].drop(i)
            plt.figure(figsize=(20, 8))
            plt.rcParams['font.size'] = 18
            sns.countplot(x=item, data=df_parameters, palette="husl").set_title('{} bar graph'.format(item))
            path = self._save_to_folder('/bar_parameters', '{}_bar_graph.pdf'.format(item))
            plt.savefig(path)

    def density_loss(self):
        sns.lineplot(self.results['iteration'], self.results['loss'], color="coral").set_title('Loss over Iteration')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        path = self._save_to_folder('/loss_graphs', 'loss.pdf')
        plt.savefig(path)

    def categorical_evolution(self, params=None):
        df_parameters = self._dataframe_parameters()
        if not params:
            params = list(df_parameters)

        df_parameters['iteration'] = self.results['iteration']
        for item in params:
            if is_float_dtype(df_parameters[item]) or is_integer_dtype(df_parameters[item]):
                continue
            else:
                for i in range(len(df_parameters[item].index)):
                    if type(df_parameters[item][i]) == str or type(df_parameters[item][i]) == bool:
                        continue
                    else:
                        df_parameters[item] = df_parameters[item].drop(i)

            # df_parameters['iteration'] = self.results['iteration']
            sns.catplot(data=df_parameters, x = 'iteration', y = item).fig.suptitle('{} over iterations'.format(item))
            path = self._save_to_folder('/category_parameters', '{}_category_iter_graph.pdf'.format(item))
            plt.savefig(path)

    def numerical_evolution(self, params=None):
        df_parameters = self._dataframe_parameters()
        if not params:
            params = list(df_parameters)
        df_parameters['iteration'] = self.results['iteration']

        for i, item in enumerate(params):
            df_parameters[item] = pd.to_numeric(df_parameters[item], errors='coerce')
            df_parameters[item].dropna(inplace=True)
            plt.figure(figsize=(20, 8))
            plt.rcParams['font.size'] = 18
            if not df_parameters[item].empty:
                sns.regplot(x = 'iteration', y = item, data=df_parameters).set_title('{} over iterations'.format(item))
                path = self._save_to_folder('/numerical_evolution', '{}_numerical_iter_graph.pdf'.format(item))
                plt.savefig(path)


visual = Visual('hyperopt_trials.csv')
visual.density_parameters()
# visual.numerical_evolution()
visual.sort_best_scores()
visual.categorical_bar_graph()
visual.categorical_evolution()
visual.density_loss()
#make an autogen for looping through all csv files
# need to make another function for looking at all of the optimization techniques comparisons --> heat map over values
# need to fix bool options in numerical numerical_evolution and density_parameters
# need to fix gamma and coef0 in numerical_evolution and density_parameters
# need to fix gamm bar graph

# import json
#
# # Save the trial results
# with open('results/trials.json', 'w') as f:
#     f.write(json.dumps(bayes_trials.results))

# # Save dataframes of parameters
# bayes_params.to_csv('results/bayes_params.csv', index = False)
# random_params.to_csv('results/random_params.csv', index = False)