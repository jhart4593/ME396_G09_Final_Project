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


# dataset
data = pd.read_csv("nanosheet_18k.csv")

# Split the data into inputs and outputs
X = data.iloc[:, :61]  # Select the first 61 columns as input data
y = data.iloc[:, 61:]  # Select the last 7 columns as output data

# Use the first 22000 entries as training data
X_train = X[1:16000]
y_train = y[1:16000]
# If you want to use the rest as testing data
X_test = X[16000:]
y_test = y[16000:]


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
print("n_estimators:", n_estimators)
print("max_depth:", max_depth)
print("learning_rate:", learning_rate)
print("min_child_weight:", min_child_weight)
print("base_score:", base_score)


t0 = time()
y_pred_best = xgb_reg_best_full.predict(X_test)
t1 = time()


print('The CD Inference time took ', t1-t0, ' seconds.')
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_best))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_best))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_best)))
print('R square:', metrics.r2_score(y_test, y_pred_best))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test, y_pred_best)) * (len(y_pred_best)-1)/(len(y_pred_best)-8-1))


rand_index = [1000,496] #random pixels - change and see the difference
print("random index: ", y_pred.shape)

# In[29]:


plt.rcParams.update({'figure.figsize': (10.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# In[31]:


fig3=plt.figure(dpi=1200)
plt.subplot(2,2,1)
plt.scatter([1,2,3,4,5,6,7], y_pred[rand_index[0]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[0]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H')
x_pos = np.arange(1,len(bars)+1)
plt.title('Full Wavelengths')
plt.xticks(x_pos, bars)
plt.ylabel('Dimension Value [nm]')
plt.legend()


plt.subplot(2,2,2)
plt.scatter([1,2,3,4,5,6,7], y_pred_best[rand_index[0]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[0]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H')
x_pos = np.arange(1,len(bars)+1)
plt.title('Important Wavelengths')
plt.xticks(x_pos, bars)
plt.legend()

plt.subplot(2,2,3)
plt.scatter([1,2,3,4,5,6,7], y_pred[rand_index[1]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[1]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H')
x_pos = np.arange(1,len(bars)+1)
plt.xticks(x_pos, bars)
plt.xlabel('Critical Dimension')
plt.ylabel('Dimension Value [nm]')
plt.legend()

plt.subplot(2,2,4)
plt.scatter([1,2,3,4,5,6,7], y_pred_best[rand_index[1]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[1]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H')
x_pos = np.arange(1,len(bars)+1)
plt.xticks(x_pos, bars)
plt.xlabel('Critical Dimension')
plt.legend()

plt.show()
