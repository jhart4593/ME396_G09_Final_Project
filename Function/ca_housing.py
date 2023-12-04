import matplotlib.pyplot as plt
import xgb_func as xgbf

# CA housing value example
from sklearn.datasets import fetch_california_housing

#pull CA housing data from sklearn - https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
housing = fetch_california_housing(as_frame = True)

# data split into input and output, get data labels
housing.data 
housing.target
data_labels = list(housing.data.columns)

# running xgboost function
[full,red,best] = xgbf.xgb_pred(housing.data,housing.target,data_labels,rand_cv=True,cv_in=2,n_iter_in=2)
plt.show()
