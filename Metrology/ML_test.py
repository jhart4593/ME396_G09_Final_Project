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

#data = scipy.io.loadmat('Hourglass_lib_LHS_30K.mat')
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
df_REFL=X[0:]
df_CD=y[0:]

#trains model
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train, y_train)

start = time()
y_pred = model.predict(X_test)
test_end = time() - start
print('The CD Inference time took ', test_end, ' seconds.')


# ### Accuracy Evaluation
#
# In this part, we evaluate following accuracy metrics:
# * Mean Absolute Error (MAE)
# * Mean Squared Error (MSE)
# * Room Mean Squared Error (RMSE)
# * R-Square
# * Adjusted R-Square

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R square:', metrics.r2_score(y_test, y_pred))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test, y_pred)) * (len(y_pred)-1)/(len(y_pred)-8-1))

model = model

# sorting thr importance scores
important_index = [str(5*x+400) for x in model.feature_importances_.argsort()[::-1][:61]]


# In[16]:


fig = plt.figure(dpi=1200) # high resolution
plt.rcParams.update({'figure.figsize': (12.0, 5.0)})
plt.rcParams.update({'font.size': 14})

plt.bar(important_index, sorted(model.feature_importances_,reverse=True))
plt.tight_layout()
plt.xticks(fontsize=12, rotation=90)
plt.title('Feature Importance Scores Obtained by XGBoost')
plt.xlabel('Wavelelngth [nm]')
plt.ylabel('Importance Score Produced by XGBoost')


sorted_wave_index = model.feature_importances_.argsort()[::-1][:61]
sorted_wave = [400+5*x for x in sorted_wave_index]

model2 = SelectFromModel(model, prefit=True)
X_new = model2.transform(X_train)
print('The method decided that only top',X_new.shape[1],
      'most informative wavelengths are sufficient for accurate CD inference.')


# Here, we create the reflectance data using down-selected wavelengths only.
important_features = model.feature_importances_.argsort()[::-1][:X_new.shape[1]]
# Here, we create the reflectance data using down-selected wavelengths only.
imp_index = [str(x) for x in important_features]
df_REFL_Reduced = df_REFL[imp_index]

# reduced -> red
X_red = df_REFL_Reduced
y_red = df_CD
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.25, random_state=101)
reduced_model = xgb.XGBRegressor()
start_red = time()
reduced_model.fit(X_train_red, y_train_red)
train_time_red = time() - start_red
print('The training time took ', train_time_red, ' seconds.')

start = time()
y_pred_red = reduced_model.predict(X_test_red)
test_end = time() - start
print('The CD Inference time took ', test_end, ' seconds.')

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test_red, y_pred_red))
print('MSE:', metrics.mean_squared_error(y_test_red, y_pred_red))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test_red, y_pred_red)))
print('R square:', metrics.r2_score(y_test_red, y_pred_red))
print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test_red, y_pred_red)) * (len(y_pred_red)-1)/(len(y_pred_red)-8-1))



rand_index = [922,120] #random pixels - change and see the difference
print("random index: ", y_pred.shape)

# In[29]:


plt.rcParams.update({'figure.figsize': (10.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# In[31]:
print(y_pred[rand_index[0]])

fig2=plt.figure(dpi=1200)
plt.subplot(2,2,1)
plt.scatter([1,2,3,4,5,6,7], y_pred[rand_index[0]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[0]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H', '')
x_pos = np.arange(1,len(bars)+1)
plt.title('Full Wavelengths')
plt.xticks(x_pos, bars)
plt.ylabel('Dimension Value [nm]')
plt.legend()


plt.subplot(2,2,2)
plt.scatter([1,2,3,4,5,6,7], y_pred_red[rand_index[0]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[0]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H','')
x_pos = np.arange(1,len(bars)+1)
plt.title('Important Wavelengths')
plt.xticks(x_pos, bars)
plt.legend()

plt.subplot(2,2,3)
plt.scatter([1,2,3,4,5,6,7], y_pred[rand_index[1]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[1]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H','')
x_pos = np.arange(1,len(bars)+1)
plt.xticks(x_pos, bars)
plt.xlabel('Critical Dimension')
plt.ylabel('Dimension Value [nm]')
plt.legend()

plt.subplot(2,2,4)
plt.scatter([1,2,3,4,5,6,7], y_pred_red[rand_index[1]],marker='x',color='red',s=30,label='Prediction')
plt.scatter([1,2,3,4,5,6,7], y_test.iloc[rand_index[1]],marker='o',color='blue',s=100,alpha=0.3,label='Original')
plt.ylim([0,120])
bars = ('A', 'B', 'C','D','E','F','H','')
x_pos = np.arange(1,len(bars)+1)
plt.xticks(x_pos, bars)
plt.xlabel('Critical Dimension')
plt.legend()

"""
score = model.score(X_train, y_train)
print("Training score: ", score)

#----------------------- Cross Validation-------------------------------------------------------------------

# - cross validataion
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
"""
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
