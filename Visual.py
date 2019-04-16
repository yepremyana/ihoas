import os
current_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
from pathlib import Path

import pandas as pd
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
        self.current_dir = os.path.join(current_dir, os.path.dirname(file) + '/Visual')

    def sort_best_scores(self):
        results = self.results

        # Sort with best scores on top and reset index for slicing
        results.sort_values('loss', ascending = True, inplace = True)
        results.reset_index(inplace = True, drop = True)
        print(results)

        return results

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
                plt.close('all')

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
                # ToDO: fix
                # this is broken
                for i in range(len(df_parameters[item].index)):
                    if type(df_parameters[item][i]) == str or type(df_parameters[item][i]) == bool:
                        continue
                    else:
                        # ToDO: enhancement
                        #this is where I could potentially make it so that an additional bar said numerical
                        df_parameters[item].drop(i)
            try:
                plt.figure(figsize=(20, 8))
                plt.rcParams['font.size'] = 18
                sns.countplot(x=item, data=df_parameters, palette="husl").set_title('{} bar graph'.format(item))
                path = self._save_to_folder('/bar_parameters', '{}_bar_graph.pdf'.format(item))
                plt.savefig(path)
                plt.close('all')
            except:
                continue

    def density_loss(self):
        plt.figure(figsize=(20, 8))
        plt.rcParams['font.size'] = 18
        sns.lineplot(self.results['iteration'], self.results['loss'], color="coral").set_title('Loss over Iteration')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        path = self._save_to_folder('/loss_graphs', 'loss.pdf')
        plt.savefig(path)
        plt.close('all')

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
            try:
                plt.figure(figsize=(20, 8))
                plt.rcParams['font.size'] = 18
                sns.catplot(data=df_parameters, x = 'iteration', y = item).fig.suptitle('{} over iterations'.format(item))
                path = self._save_to_folder('/category_evolution', '{}_category_iter_graph.pdf'.format(item))
                plt.savefig(path)
                plt.close('all')
            except:
                continue

    def numerical_evolution(self, params=None):
        df_parameters = self._dataframe_parameters()
        if not params:
            params = list(df_parameters)

        for i, item in enumerate(params):
            df_parameters[item] = pd.to_numeric(df_parameters[item], errors='coerce')
            df_item = pd.concat([df_parameters[item], self.results['iteration']], axis = 1)
            df_item.dropna(axis = 0, how = 'any',inplace=True)
            plt.figure(figsize=(20, 8))
            plt.rcParams['font.size'] = 18
            if not df_item.empty:
                sns.regplot(data=df_item, x = 'iteration', y = item).set_title('{} over iterations'.format(item))
                path = self._save_to_folder('/numerical_evolution', '{}_numerical_iter_graph.pdf'.format(item))
                plt.savefig(path)
                plt.close('all')

    def plot_all(self):
        self.density_parameters()
        self.numerical_evolution()
        self.categorical_bar_graph()
        self.categorical_evolution()
        self.density_loss()

# ToDO:
# need to make another function for looking at all of the optimization techniques comparisons --> heat map over values
# need to fix bool options in numerical numerical_evolution and density_parameters
# need to fix gamma bar graph, count when it picked the auto vs numerical choice in bar graph.
