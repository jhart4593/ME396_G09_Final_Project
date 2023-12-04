import matplotlib.pyplot as plt
import pandas as pd
import xgb_func as xgbf

# metrology example
dataset = pd.read_csv("nanosheet_18k.csv")

# Split the data into inputs and outputs
input_data = dataset.iloc[:, :61]  # Select the first 61 columns as input data

target_data = dataset.iloc[:, 61:]  # Select the last 7 columns as output data

# determine data labels
data_labels = list(input_data.columns)

# run xgboost function - does not return plots when target data has multiple columns
[full,red,best] = xgbf.xgb_pred(input_data,target_data,data_labels,rand_cv=True,cv_in=2,n_iter_in=2)
plt.show()