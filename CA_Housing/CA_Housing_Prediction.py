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


# --------------------- Full Model-------------------------------------------------------------------------------

#split data inputs and output into training and testing data (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.25, random_state=101)

#trains model
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

start = time()
y_pred = model.predict(X_test)
test_end = time() - start
print('The Median House Value Inference time for the full model took ', test_end, ' seconds.')
print('')

# ### Accuracy Evaluation
#
# In this part, we evaluate following accuracy metrics:
# * Mean Absolute Error (MAE)
# * Mean Squared Error (MSE)
# * Room Mean Squared Error (RMSE)
# * R-Square
# * Adjusted R-Square

from sklearn import metrics
print('Full Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R square:', metrics.r2_score(y_test, y_pred))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test, y_pred)) * (len(y_pred)-1)/(len(y_pred)-8-1))
print('')

# sorting the importance scores
important_index = model.feature_importances_.argsort()[::-1]

# plot importance scores
plt.figure(1)
plt.bar(list(housing.data.columns), model.feature_importances_)
plt.tight_layout()
plt.xticks(fontsize=12, rotation=90)
plt.title('Feature Importance Scores Obtained by XGBoost')
plt.xlabel('CA Housing Attribute')
plt.ylabel('Importance Score Produced by XGBoost')

# --------------------Reduced Model----------------------------------------------------------------------

# sklearn selects most important features from the model
model2 = SelectFromModel(model, prefit=True)
X_new = model2.transform(X_train)
print('The method decided that only top',X_new.shape[1],
      'most informative housing attributes are sufficient for accurate median house value inference.')

# Here, we get the feature data for the down-selected housing attributes only.
# reduced --> red
important_features = model.feature_importances_.argsort()[::-1][:X_new.shape[1]]
X_red = housing.data.iloc[:,important_features]
y_red = housing.target

#train reduced model
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.25, random_state=101)
reduced_model = xgb.XGBRegressor(objective ='reg:squarederror')
start_red = time()
reduced_model.fit(X_train_red, y_train_red)
train_time_red = time() - start_red
print('The training time took ', train_time_red, ' seconds.')

start = time()
y_pred_red = reduced_model.predict(X_test_red)
test_end = time() - start
print('The Median House Value Inference time for the reduced model took ', test_end, ' seconds.')
print('')

# Accuracy evaluation
from sklearn import metrics
print('Reduced Model Metrics')
print('MAE:', metrics.mean_absolute_error(y_test_red, y_pred_red))
print('MSE:', metrics.mean_squared_error(y_test_red, y_pred_red))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_red, y_pred_red)))
print('R square:', metrics.r2_score(y_test_red, y_pred_red))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test_red, y_pred_red)) * (len(y_pred_red)-1)/(len(y_pred_red)-8-1))
print('')

# plot prediction vs actual values for all test data
plt.figure(2)
plt.scatter(np.arange(len(y_test)),abs(y_pred-y_test),label='Absolute Error')
plt.axhline(y=metrics.mean_absolute_error(y_test, y_pred),color='r',label='Mean Absolute Error')
plt.xlabel('CA Housing Data Point')
plt.ylabel('Absolute Error ($100k)')
plt.title('Full Model Prediction vs Actual')
plt.legend()

plt.figure(3)
plt.scatter(np.arange(len(y_test_red)),abs(y_pred_red-y_test_red),label='Absolute Error')
plt.axhline(y=metrics.mean_absolute_error(y_test_red, y_pred_red),color='r',label='Mean Absolute Error')
plt.xlabel('CA Housing Data Point')
plt.ylabel('Absolute Error ($100k)')
plt.title('Reduced Model Prediction vs Actual')
plt.legend()


#-------------------------------Cross-Validation----------------------------------------------------------------------------

#tuning hyperparameters
n_estimators = [350, 400, 500, 550, 600]
max_depth = [5,10,15,20]
learning_rate = [0.05, 0.1, 0.2, 0.25]
min_child_weight = [4,5,6,7,8]
base_score = [0.75, 1,1.5,2,2.5]

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'min_child_weight': min_child_weight,
    'base_score': base_score
}

random_cv = RandomizedSearchCV(estimator=model,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5,
            return_train_score = True,
            random_state=42)


random_cv.fit(X_train, y_train)
random_cv.best_estimator_.save_model('Best_Model_Full.json')
random_cv.best_estimator_.save_model('Best_Model_Full.txt')

# Save the best model using pickle
with open("Best_Model_Full.pkl", "wb") as file:
    pickle.dump(random_cv.best_estimator_, file)

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
