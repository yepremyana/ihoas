# ihoas
Informative hyperparameter optimization and selection

## Table of contents
* [General info](#general-info)
* [Dependencies](#dependencies)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## General info
  Hyper-parameter optimization methods allow efficient and robust hyperparameter searching without the need 
  to hand select each value and combination. Although hyper-parameter tuners, such as BOHB, Hyperopt and SMAC,
  have been investigated by researchers in terms of performance, there has yet to be a deep analysis on the values 
  each tuner selected over all iterations. We propose a thorough aggregation of data in terms of the efficiency of the search 
  values selected by each tuner over 59 datasets and 10 popular ML algorithms from Scikit-learn. From this large data 
  accumulated, we observe and advise which tuners show better results for certain datasets, through its meta-data, and 
  algorithms. Through this research, we have also developed a simple system for easy implementation of various tuners in one 
  repository. A user can compare 5 different optimation techniques in one single repository: GridSearch, SMAC, BOHB, HB, 
  and hyperopt.

## Dependencies
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following dependencies.

```bash
pip install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@dev-dist#egg=sklearn_wrap
pip install git+https://gitlab.com/datadrivendiscovery/d3m.git@devel
pip install Jinja2==2.9.4
pip install hyperopt
pip install smac
pip install simplejson==3.12.0
pip install hpbandster
```

## Usage
```bash
python autogen_thesis.py --problem_type classification
```

## Authors
Alice Yepremyan

## Inspiration for Visuals
https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a

## Acknowledgements
