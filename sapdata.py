#!/usr/bin/env python
# coding: utf-8

# Check if all the required libraries are already installed and available in your environment

pip freeze


get_ipython().run_line_magic('pip', 'install sklearn-pandas')

pip install web.py

# Install sklearn_pandas library
%pip install sklearn-pandas


# Import all the libraries required
import pandas as pd
import numpy as np
import scipy.stats
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder

# get rid of warnings
import warnings
warnings.filterwarnings("ignore")
# get more than one output per Jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# for functions we implement later
from utils import best_fit_distribution
from utils import plot_result

import json
df= [json.loads(line) for line in open('imagesh_htl67e00_2063610012', 'r')]
print(df)

import numpy as np
X = np.array([
        df
    ])
X.shape


for c in categorical:
        counts = df[c].value_counts()
        np.random.choice(list(counts.index), p=(counts/len(df)).values, size=5)
        
print(df)      


for c in continuous:
    data = df[c]
    best_fit_name, best_fit_params = best_fit_distribution(data, 50)
    best_distributions.append((best_fit_name, best_fit_params))
print(data)


def generate_like_df(df, categorical_cols, continuous_cols, best_distributions, n, seed=0):
    np.random.seed(seed)
    d = {}

    for c in categorical_cols:
        counts = df[c].value_counts()
        d[c] = np.random.choice(list(counts.index), p=(counts/len(df)).values, size=n)

    for c, bd in zip(continuous_cols, best_distributions):
        dist = getattr(scipy.stats, bd[0])
        d[c] = dist.rvs(size=n, *bd[1])

    return pd.DataFrame(d, columns=categorical_cols+continuous_cols)


gendf = generate_like_df(df, categorical, continuous, best_distributions, n=100)
gendf.shape
gendf.head()

gendf.columns = list(range(gendf.shape[1]))


gendf.to_csv("output.csv", index_label="id")


def plot_result(df, continuous, best_distributions):
    for c, (best_fit_name, best_fit_params) in zip(continuous, best_distributions):
        best_dist = getattr(st, best_fit_name)
        pdf = make_pdf(best_dist, best_fit_params)
        _ = plt.figure(figsize=(12,8))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        _ = df[c].plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join([f'{k}={v:0.2f}' for k,v in zip(param_names, best_fit_params)])
        dist_str = f'{best_fit_name}({param_str})'
        _ = ax.set_title(c+ " " + dist_str)
        _ = ax.set_ylabel('Frequency')
        plt.show();




