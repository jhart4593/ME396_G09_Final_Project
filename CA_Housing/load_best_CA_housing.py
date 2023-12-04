import xgboost as xgb
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import scipy.io
from time import time
import json
import pickle


from sklearn.datasets import fetch_california_housing

#pull CA housing data from sklearn - https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
housing = fetch_california_housing(as_frame = True)

#split data inputs and output into training and testing data (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.25, random_state=101)

# To load the model back:
with open("Best_Model_Full.pkl", "rb") as file:
    xgb_reg_best_full = pickle.load(file)

"""
xgb_reg_best_full = xgb.XGBRegressor()
xgb_reg_best_full.load_model("Best_Model_Full.json")
"""

# Access the hyperparameters
n_estimators = xgb_reg_best_full.n_estimators
max_depth = xgb_reg_best_full.max_depth
learning_rate = xgb_reg_best_full.learning_rate
min_child_weight = xgb_reg_best_full.min_child_weight
base_score = xgb_reg_best_full.base_score

# Print the hyperparameters
print('Best Model Hyperparameters')
print("n_estimators:", n_estimators)
print("max_depth:", max_depth)
print("learning_rate:", learning_rate)
print("min_child_weight:", min_child_weight)
print("base_score:", base_score)
print('')


t0 = time()
y_pred_best = xgb_reg_best_full.predict(X_test)
t1 = time()


print('The Median House Value Inference time for the best model took ', t1-t0, ' seconds.')
print('')

from sklearn import metrics
print('Best Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_best))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_best))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_best)))
print('R square:', metrics.r2_score(y_test, y_pred_best))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test, y_pred_best)) * (len(y_pred_best)-1)/(len(y_pred_best)-8-1))

# plot prediction vs actual values for all test data
plt.figure(4)
plt.scatter(np.arange(len(y_test)),abs(y_pred_best-y_test),label='Absolute Error')
plt.axhline(y=metrics.mean_absolute_error(y_test, y_pred_best),color='r',label='Mean Absolute Error')
plt.xlabel('CA Housing Data Point')
plt.ylabel('Absolute Error ($100k)')
plt.title('Best Model Prediction vs Actual')
plt.legend()

plt.show()