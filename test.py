import pickle
from numpy import *

print('Loading XGBoost model')
filename = 'final_xgb.sav'
gbm = pickle.load(open(filename, 'rb'))

print('Loading XGBoost x_test')
filename = 'xgb_test.sav'
x_xgb_test = pickle.load(open(filename, 'rb'))

print('Loading Random Forest classifier model')
filename = 'final_rf.sav'
RFC = pickle.load(open(filename, 'rb'))

print('Loading Random Forest x_test')
filename = 'rf_test.sav'
x_test = pickle.load(open(filename, 'rb'))

print('Loading y_test')
filename = 'y_test.sav'
y_test = pickle.load(open(filename, 'rb'))

import xgboost as xgb

print('Predicting')
y_hat_rf = RFC.predict(x_test)
y_hat_xgb = gbm.predict(xgb.DMatrix(x_xgb_test))

y_hat_ensemble = y_hat_rf * y_hat_xgb

accuracy = mean(y_hat_ensemble == y_test)
accuracy_rf = mean(y_hat_rf == y_test)
accuracy_xgb = mean(y_hat_xgb == y_test)

print('Accuracy XGBoost: '+str(accuracy_xgb))
print('Accuracy Random Forest: '+str(accuracy_rf))
print('\nFinal model (RF+XGB logical AND ensemble) accuracy : '+str(accuracy))