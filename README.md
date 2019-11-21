# Simple Way to Anonymize Data with Python and Pandas
Recently, I got into a project challenge that requires one to build a machine learning algorithm
that will be able to anonymize invoices' personal information data. 


The dataset contained sensitive personal information like names of the clients, signatures, and
handwritten notes that needed to be anonymized. The goal was to let the data remain effective even
after training the data by preserving the style, the orientation, complexities, and the imperfections
of the original data. 


The anonymized data was supposed to remain simpler and clearer when one reads it as compared to the
original data. So I did a short script that will perform the tasks and change the dataset while still
preserving the same key information. 


Below, I will detail every step that I took, together with the [AWS] (https://ap-south-1.console.aws.amazon.com/console/home?region=ap-south-1#/) and [Github] (https://github.com/ednasawe/Simple-Way-to-Anonymize-Data-with-Python-and-Pandas/blob/master/README.md/) links for the source code and my [blogpost] (https://techauthorityorg.wordpress.com/) to highlight the journey and steps. 


Prepare and develop a system to replace the content into an anonymized dataset that uses a machine-learning
algorithm such that the personal identifying information is removed and also the anonymized data remains
effective training dataset. 


For this task, I worked with a dataset of 25,000 invoices and the invoice like documents from the RVL-CDIP Dataset
and the ground truth labels of 1,000 labelled invoices from the dataset.


To start, we will be using a file that contains only three images that must be anonymized before using the other
large dataset. The zip folder with the three images contains a PNG file, a PNG file with the bounding boxes and
lastly a JSON file with the coordinates of the bounding boxes. So, we will start with the three images first ...

### I will be using Jupyter notebook as my environment. First, let us import all the required libraries:

```
import pandas as pd
import numpy as np
import scipy.stats
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
```

I will assume that you are familiar with the libraries that we have imported above. For instance, the sklearn_pandas
is a conventional library that tries to bridge the gap between two packages. It gets to provide a DataFrameMapper
class that we will be using in making working with pandas DataFrames much easier to allow easy changing of the
encoding of the variables in the code.


The IPython.core.interactiveshell helps to display more that one output. More on Jupyter cheatsheet can be found
in this blog post. Lastly, we will be putting some of the code into a file called utils.pu. This will be placed
next to the notebook.


```df = pd.read_csv("../desktop/SAP Data Analytics ML Challenge/leaderboard_task2") ```


For the analysis step, we will use the three dataset images:

```
df.shape
df.head()
```


Output:






