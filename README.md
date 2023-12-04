# ME396_G09_Final_Project

## Predicting Large Datasets Using XGBoost
James Wang, Justin Hart, Mobina Tavangarifard

We demonstrate the use of a decision tree ML algorithm in predicting large datasets using the XGBoost library.

## Links
Project Report: https://docs.google.com/document/d/1RXT-DNYlY3WjcMRRzVr92ZcNn37_6c_yp09cDTvRVfE/edit?usp=sharing 
Project Presentation: https://docs.google.com/presentation/d/1XiW7GZcUZVbdhYx3ifEzNfSAGtFmcoyo7fH9i1bOLlc/edit?usp=sharing

## Running the Code
### Packages to Install:
XBGoost - https://xgboost.readthedocs.io/en/stable/install.html#python 
Numpy - https://numpy.org/install/ 
Sklearn - https://scikit-learn.org/stable/install.html 
Matplotlib - https://matplotlib.org/stable/users/installing/index.html 
Pandas - https://pandas.pydata.org/docs/getting_started/install.html 

### Example Datasets:
CA Housing:
- In CA_Housing folder on the github repo, open CA_Housing_Prediction.py
  - Run this file to see results of predicting a dataset on California median house values
  - To avoid long run times, comment out code for parameter optimized model (best model). This is everything past line 126.
    - To see results of parameter optimized model, run load_best_CA_Housing.py which runs a saved best model
      
Metrology:
In Metrology folder on the github repo, open ML_test.py
  - Ensure nanosheet_18k.csv is located in the same folder as ML_test.py
  - Run this file to see results of predicting a dataset on metrology measurements
  - To avoid long run times, comment out code for parameter optimized model (best model). This is everything past line 182.
    - To see results of parameter optimized model, run load_best_Metrology.py which runs a saved best model

Function:
- In the Function folder of the github repo, the xgb_func.py file contains a function xgb_pred that can be used to create full, reduced, and best trained models for a set   
  of given input data
  - Inputs: data from dataset to be used in prediction, target from dataset which is the variable to be predicted, column labels of data from dataset, testing data split 
    percentage, True/False to perform parameter optimization to find the best model, dictionary of parameters and range of their values to be optimized over, number of 
    folds in cross-validated parameter optimization, number of parameter settings sampled for parameter optimization
  - Outputs: full, reduced, and best trained models
- The metrology.py, ca_housing.py, and life_expectancy.py, are three scripts with example usage of the xgb_pred function on metrology, California median housing values, and 
  life expectancy data respectively
- Run each of these scripts to see results of predicting a dataset with their respective data using the xgb_pred function.
  - Ensure life_exp_data.csv and nanosheet_18k.csv are located in the same folder as the example scripts
  - Set rand_cv argument to ‘False’ in order to reduce run time by eliminating parameter optimization for best model (will only return full and reduced models in this case)

