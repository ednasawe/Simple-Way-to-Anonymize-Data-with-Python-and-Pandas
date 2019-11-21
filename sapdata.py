#!/usr/bin/env python
# coding: utf-8

# Check if all the required libraries are already installed and available in your environment
pip freeze

# Install sklearn_pandas library
%pip install sklearn-pandas

# Import all the libraries required
import pandas as pd
import numpy as np
import scipy.stats
import json
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder

# Open and read the json file
openfile=open('imagesh_htl67e00_2063610012.json')
jsondata=json.load(openfile)
df=pd.DataFrame(jsondata)

openfile.close()
print(df)

# Display the json file in column
df.shape
df.head()
