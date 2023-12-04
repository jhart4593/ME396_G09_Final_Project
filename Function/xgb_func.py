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

# default parameters and values for function
base_params = {'n_estimators':[350,400,500,550,600],'max_depth':[5,10,15,20],
               'learning_rate':[0.05,0.1,0.2,0.25],'min_child_weight':[4,5,6,7,8],
               'base_score':[0.75,1,1.5,2,2.5]}


def xgb_pred(data,target,data_labels,test_split=0.25,rand_cv=False,params=base_params,cv_in=5,n_iter_in=10):
    """
    Uses xgboost to train 3 different regression models using an input dataset.
    
    Args:
        data (dataframe): input data used to predict the target data
        target (dataframe): output data that is predicted from the input data
        data_labels (list): column labels of input data (data)
        test_split (float): percentage of data to be used for testing - default is 25%
        rand_cv (bool): True to perform parameter optimization in order to find the best model - default is False
        params (dictionary): dictionary of parameters and test values for parameter optimization.
                             There is a default dictionary set (base_params)
        cv_in (int): number of folds in cross-validator parameter optimization - default is 5, minimum is 2
        iter_in (int): number of parameter settings sampled for parameter optimization - default is 10, minimum is 2
    
    Returns:
        Full, reduced, and best prediction models if rand_cv=True
        Full, reduced models if rand_cv=False
    """

    # --------------------- Full Model-------------------------------------------------------------------------------

    #split data inputs and output into training and testing data (75% training, 25% testing)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_split, random_state=101)

    #trains model
    model = xgb.XGBRegressor(objective ='reg:squarederror')
    start_full_train = time()
    model.fit(X_train, y_train)
    train_time_full = time()-start_full_train
    print('The training time for the full model took ',train_time_full,' seconds.')

    # get time for testing
    start = time()
    y_pred = model.predict(X_test)
    test_end = time() - start
    print('The Inference time for the full model took ', test_end, ' seconds.')
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
    plt.bar(data_labels, model.feature_importances_)
    plt.tight_layout()
    plt.xticks(fontsize=12, rotation=90)
    plt.title('Feature Importance Scores Obtained by XGBoost')
    plt.xlabel('Data')
    plt.ylabel('Importance Score Produced by XGBoost')

    # --------------------Reduced Model----------------------------------------------------------------------

    # sklearn selects most important features from the model
    model2 = SelectFromModel(model, prefit=True)
    X_new = model2.transform(X_train)
    print('The method decided that only top',X_new.shape[1],
          'most informative data attributes are sufficient for accurate target inference.')

    # Here, we get the feature data for the down-selected housing attributes only.
    # reduced --> red
    important_features = model.feature_importances_.argsort()[::-1][:X_new.shape[1]]
    X_red = data.iloc[:,important_features]
    y_red = target

    #train reduced model
    X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=test_split, random_state=101)
    reduced_model = xgb.XGBRegressor(objective ='reg:squarederror')
    start_red = time()
    reduced_model.fit(X_train_red, y_train_red)
    train_time_red = time() - start_red
    print('The training time for the reduced model took ', train_time_red, ' seconds.')

    # get time for testing
    start = time()
    y_pred_red = reduced_model.predict(X_test_red)
    test_end = time() - start
    print('The Inference time for the reduced model took ', test_end, ' seconds.')
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

    # if multiple targets skip plotting
    if target.ndim == 1:
        # plot prediction vs actual values for all test data
        plt.figure(2)
        plt.scatter(np.arange(len(y_test)),abs(y_pred-y_test),label='Absolute Error')
        plt.axhline(y=metrics.mean_absolute_error(y_test, y_pred),color='r',label='Mean Absolute Error')
        plt.xlabel('Data Point')
        plt.ylabel('Absolute Error')
        plt.title('Full Model Prediction vs Actual')
        plt.legend()

        plt.figure(3)
        plt.scatter(np.arange(len(y_test_red)),abs(y_pred_red-y_test_red),label='Absolute Error')
        plt.axhline(y=metrics.mean_absolute_error(y_test_red, y_pred_red),color='r',label='Mean Absolute Error')
        plt.xlabel('Data Point')
        plt.ylabel('Absolute Error')
        plt.title('Reduced Model Prediction vs Actual')
        plt.legend()
    else:
        print('Multiple target dataset does not produce plots')
        print('')

    #-------------------------------Cross-Validation----------------------------------------------------------------------------

    # complete random cross-validated search over parameters to find best model - parameter optimization (if designated in function input)
    if rand_cv:
        
        # set hyperparameters and cross-validation inputs
        hyperparameter_grid = params

        random_cv = RandomizedSearchCV(estimator=model,
                    param_distributions=hyperparameter_grid,
                    cv=cv_in, n_iter=n_iter_in,
                    scoring = 'neg_mean_absolute_error',n_jobs = 4,
                    verbose = 5,
                    return_train_score = True,
                    random_state=42)

        # train and save the cross-validated model
        start_best = time()
        random_cv.fit(X_train, y_train)
        train_time_best = time()-start_best
        print('The training time for the best model took ',train_time_best,' seconds.')
        random_cv.best_estimator_.save_model('Best_Model_Full.json')

        # Save the best model using pickle
        with open("Best_Model_Full.pkl", "wb") as file:
            pickle.dump(random_cv.best_estimator_, file)

        # To load the model back:
        with open("Best_Model_Full.pkl", "rb") as file:
            xgb_reg_best_full = pickle.load(file)

        # Access the hyperparameters
        best_params = {}
        for param in params.keys():
            best_params[param] = getattr(xgb_reg_best_full,param)

        # Print the hyperparameters
        print('Best Model Hyperparameters')
        for keys,values in best_params.items():
            print(f'{keys}: {values}')


        # get the testing time
        t0 = time()
        y_pred_best = xgb_reg_best_full.predict(X_test)
        t1 = time()
        print('The Inference time for the best model took ', t1-t0, ' seconds.')
        print('')

        # accuracy evaluation
        from sklearn import metrics
        print('Best Model Metrics')
        print('MAE:', metrics.mean_absolute_error(y_test, y_pred_best))
        print('MSE:', metrics.mean_squared_error(y_test, y_pred_best))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_best)))
        print('R square:', metrics.r2_score(y_test, y_pred_best))
        print('Adjusted R square:', 1 - (1-metrics.r2_score(y_test, y_pred_best)) * (len(y_pred_best)-1)/(len(y_pred_best)-8-1))
        
        #if multiple targets skip plotting
        if target.ndim == 1:
            # plot prediction vs actual values for all test data
            plt.figure(4)
            plt.scatter(np.arange(len(y_test)),abs(y_pred_best-y_test),label='Absolute Error')
            plt.axhline(y=metrics.mean_absolute_error(y_test, y_pred_best),color='r',label='Mean Absolute Error')
            plt.xlabel('Data Point')
            plt.ylabel('Absolute Error')
            plt.title('Best Model Prediction vs Actual')
            plt.legend()
        else:
            print('Multiple target dataset does not produce plots')
            print('')
    
    else:
        print('Parameter Optimized Model was not produced')
    
    if rand_cv:
        return model,reduced_model,xgb_reg_best_full
    else:
        return model,reduced_model
    

