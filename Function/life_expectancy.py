import matplotlib.pyplot as plt
import pandas as pd
import xgb_func as xgbf

# Life expectancy example
dataset = pd.read_csv("life_exp_data.csv")

# target data
target_data = dataset['Life expectancy ']

# remove target data as well as categorical data for the input (xgboost can't handle categorical data in scikit learn interface for the XGBregressor model yet)
input_data = dataset.drop(['Life expectancy ','Country','Status'],axis=1)

#get data labels
data_labels = list(input_data.columns)

# fill the nan cells with the median value of respective column
input_data_full = input_data.fillna(input_data.median())
target_data_full = target_data.fillna(target_data.median())

# run xgboost function
[full,red,best] = xgbf.xgb_pred(input_data_full,target_data_full,data_labels,rand_cv=True,cv_in=2,n_iter_in=2)
plt.show()