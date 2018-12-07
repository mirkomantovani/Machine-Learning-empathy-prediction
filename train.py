import pandas as pd
import numpy as np
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
import pickle


df = pd.read_csv('./input/responses.csv')

print('\n---------- DATA PREPROCESSING ----------\n')
# Renaming some cols
df.rename(columns={'Left - right handed': 'LeftRightHanded', 'Village - town': 'VillageTown', 'Internet usage':
    'InternetUsage', 'Only child': 'OnlyChild', 'House - block of flats': 'HouseFlats'}, inplace=True)

all_columns = list(df)

print('Dropping rows with Empathy = NaN')
df.dropna(subset=['Empathy'], inplace=True)
print('New dataset size:'+str(df.shape))

print('Missing values imputation with mode')
df[all_columns] = df[all_columns].fillna(df.mode().iloc[0])
float_cols = df.dtypes == np.float64
fake_float = list(df.loc[:, float_cols])
df[fake_float] = df[fake_float].astype(int)

print('Categorical values -> binary, ordinal, OHE')
categorical = df.dtypes == np.object
categorical_cols = list(df.loc[:, categorical])


def OHE_Smoking_N(value):
    return 1 if value == 'never smoked' else 0


def OHE_Smoking_T(value):
    return 1 if value == 'tried smoking' else 0


def OHE_Smoking_F(value):
    return 1 if value == 'former smoker' else 0


def OHE_Smoking_C(value):
    return 1 if value == 'current smoker' else 0


df['Smoking_Never'] = df.Smoking.apply(OHE_Smoking_N)
df['Smoking_Tried'] = df.Smoking.apply(OHE_Smoking_T)
df['Smoking_Former'] = df.Smoking.apply(OHE_Smoking_F)
df['Smoking_Current'] = df.Smoking.apply(OHE_Smoking_C)


def OHE_Lying_N(value):
    return 1 if value == 'never' else 0


def OHE_Lying_S(value):
    return 1 if value == 'sometimes' else 0


def OHE_Lying_O(value):
    return 1 if value == 'only to avoid hurting someone' else 0


def OHE_Lying_E(value):
    return 1 if value == 'everytime it suits me' else 0


df['Lying_Never'] = df.Lying.apply(OHE_Lying_N)
df['Lying_Sometimes'] = df.Lying.apply(OHE_Lying_S)
df['Lying_Only'] = df.Lying.apply(OHE_Lying_O)
df['Lying_Everytime'] = df.Lying.apply(OHE_Lying_E)

def OHE_Alcohol_D(value):
    return 1 if value=='drink a lot' else 0

def OHE_Alcohol_S(value):
    return 1 if value=='social drinker' else 0

def OHE_Alcohol_N(value):
    return 1 if value=='never' else 0

df['Alcohol_Drink'] = df.Alcohol.apply(OHE_Alcohol_D)
df['Alcohol_Social'] = df.Alcohol.apply(OHE_Alcohol_S)
df['Alcohol_Never'] = df.Alcohol.apply(OHE_Alcohol_N)

def OHE_Male(value):
    return 1 if value=='male' else 0

df['Male'] = df.Gender.apply(OHE_Male)

def OHE_Right(value):
    return 1 if value=='right handed' else 0

df['Right_Handed'] = df.LeftRightHanded.apply(OHE_Right)

def OHE_Only(value):
    return 1 if value=='yes' else 0

df['Only_Child'] = df.OnlyChild.apply(OHE_Only)

def OHE_City(value):
    return 1 if value=='city' else 0

df['City'] = df.VillageTown.apply(OHE_City)

def OHE_House(value):
    return 1 if value=='house/bungalow' else 0

df['House'] = df.HouseFlats.apply(OHE_House)


def OHE_Punctuality_Ordinal(value):
    if value=='i am always on time':
        return 1
    if value=='i am often early':
        return 2
    if value=='i am often running late':
        return 0
    #should not arrive here
    return 0

df['Punctual'] = df.Punctuality.apply(OHE_Punctuality_Ordinal)


def OHE_InternetUsage_Ordinal(value):
    if value=='most of the day':
        return 3
    if value=='few hours a day':
        return 2
    if value=='less than an hour a day':
        return 1
    if value=='no time at all':
        return 0
    #should not arrive here
    return 0

df['InternetUsage'] = df.InternetUsage.apply(OHE_InternetUsage_Ordinal)


def OHE_Education_Ordinal(value):
    if value=='doctorate degree':
        return 5
    if value=='masters degree':
        return 4
    if value=='college/bachelor degree':
        return 3
    if value=='secondary school':
        return 2
    if value=='primary school':
        return 1
    if value=='currently a primary school pupil':
        return 0
    #should not arrive here
    return 0

df['Education'] = df.Education.apply(OHE_Education_Ordinal)

encoded_columns = ['Smoking','Lying','Alcohol','Gender','LeftRightHanded','OnlyChild','VillageTown','HouseFlats','Punctuality']

df = df.drop(encoded_columns, axis=1)

print('\n---------- BASELINE AND SIMPLE CLASSIFIERS ----------\n')
print('Logistic Regression multiclass (1,2,3,4,5) and then converting to 0-1')
pivot = int(0.8 * len(df))
predicting_features = list(df)
predicting_features.remove('Empathy')

x = np.array(df[predicting_features])
y = np.array(df['Empathy'])

x_test = x[pivot:]
x_train = x[:pivot]

y_test = y[pivot:]
y_train = y[:pivot]

LR = LogisticRegression()
LR.fit(x_train, y_train)
y_hat = LR.predict(x_test)

y_hat_train = LR.predict(x_train)

def transform_in_binary(empathy_level):
    return 0 if empathy_level < 4 else 1

bin_trans = vectorize(transform_in_binary)

y_hat_binary = bin_trans(y_hat)
y_test_binary = bin_trans(y_test)

y_hat_train_binary = bin_trans(y_hat_train)
y_train_binary = bin_trans(y_train)

print('Accuracy on Test '+str(mean(y_hat_binary == y_test_binary))+'Accuracy on Train '+str(mean(y_hat_train_binary == y_train_binary)))

print('SVM OVO multiclass (1,2,3,4,5) and then converting to 0-1')

svm_ovo = svm.SVC(C=1.0, decision_function_shape='ovo')
svm_ovo.fit(x_train, y_train)

y_hat = svm_ovo.predict(x_test)

y_hat_binary = bin_trans(y_hat)
y_test_binary = bin_trans(y_test)

y_hat_train = svm_ovo.predict(x_train)

print('Accuracy on Test '+str(mean(y_hat_binary == y_test_binary))+'Accuracy on Train '+str(mean(y_hat_train == y_train)))

print('Linear SVM binary classifier')

svm_linear = svm.SVC()
svm_linear.fit(x_train, y_train_binary)

y_hat_binary = svm_linear.predict(x_test)

print('Accuracy on Test '+str(mean(y_hat_binary == y_test_binary)))

print('BASELINE: majority voting classifier')
print('Accuracy: '+str(mean(concatenate((y_train_binary,y_test_binary), axis=None))))

print('\n---------- COMPLEX MODELS, ensembles ----------\n')
print('Random forest regressor and conversion to 0-1')
selected_features = predicting_features
x = np.array(df[selected_features])
y = np.array(df['Empathy'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
RFR = RandomForestRegressor(n_estimators=300, n_jobs=-1, verbose=0)
RFR.fit(x_train, y_train)
y_hat = RFR.predict(x_test)
def transform_contininous_in_binary(empathy_level):
    return 0 if empathy_level < 3.5 else 1
cont_bin_trans = vectorize(transform_contininous_in_binary)
y_hat_binary = cont_bin_trans(y_hat)
y_test_binary = cont_bin_trans(y_test)
y_hat_train = RFR.predict(x_train)
y_hat_train_binary = cont_bin_trans(y_hat_train)
y_train_binary = cont_bin_trans(y_train)
print('Accuracy on Test '+str(mean(y_hat_binary == y_test_binary))+'Accuracy on Train '+str(mean(y_hat_train_binary == y_train_binary)))

print('Random forest binary classifier 20-fold CV')
def encode_emp(value):
    return 0 if value < 4 else 1

df['Empathy'] = df.Empathy.apply(encode_emp)

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

kf = KFold(n_splits=20)
x = np.array(df[selected_features])
y = np.array(df['Empathy'])
kf.get_n_splits(x)

accuracy = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    RFR = RandomForestClassifier(n_estimators=50, n_jobs=-1, verbose=0)

    RFR.fit(x_train, y_train)
    y_hat = RFR.predict(x_test)
    accuracy.append(mean(y_hat == y_test))

print('Accuracy '+str(mean(accuracy)))

print('\n---------- BUILDING FINAL MODEL ----------\n')
print('Train/dev/set split: 80/20 and crossvalidation to tune Hyperparameters')

x = df[selected_features].copy()
y = df['Empathy'].copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)
print('\n---------- Parameters tuning and feature selection ----------\n')
print('Parameters tuning Random Forest using crossvalidation and validationcurve')

params = range(1, 111, 5)

scores, tst_scr = validation_curve(RandomForestClassifier(), x_train,\
                                       y_train, 'n_estimators', params, \
                                       cv=5,n_jobs=-1, verbose=1)

tst_scr_mean = tst_scr.mean(axis=1)
index_min = np.argmax(tst_scr_mean)
best_n_estimators = params[index_min]
print('Best n_estimators: '+str(best_n_estimators)+', accuracy:'+str(max(tst_scr_mean)))

params = range(5, 20, 1)

scores, tst_scr = validation_curve(RandomForestClassifier(n_estimators = 100), x_train,\
                                       y_train, 'max_depth', params, \
                                       cv=20,n_jobs=-1, verbose=1)
tst_scr_mean = tst_scr.mean(axis=1)
index_min = np.argmax(tst_scr_mean)
best_max_depth = params[index_min]
print('Best max_depth: '+str(best_max_depth)+', accuracy:'+str(max(tst_scr_mean)))

params = range(6, 14, 1)

scores, tst_scr = validation_curve(RandomForestClassifier(n_estimators= 100, max_depth = 14), x_train,\
                                       y_train, 'max_features', params, \
                                       cv=10,n_jobs=-1, verbose=1)

tst_scr_mean = tst_scr.mean(axis=1)
index_min = np.argmax(tst_scr_mean)
best_max_features = params[index_min]
print('Best max_features: '+str(best_max_features)+', accuracy:'+str(max(tst_scr_mean)))

print('\nFeature selection, using SelectKBest, for XGboost (since Boosting does not work well with a lot of features, especially if they are correlated)\n')

# Perform feature selection
selector = SelectKBest(f_regression, k=50)
selector.fit(df[selected_features], df['Empathy'])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

sc = list(zip(selected_features,scores))
sc.sort(key=lambda tup: tup[1])

print('The most important 20 features that will be used for XGBoost:')
sc.sort(key=lambda tup: tup[1])  # sorts in place
a = sc[137:]
new_feat = [x for (x,y) in a]
new_feat.reverse()
new_feat

print(new_feat)

print('Parameters tuning XGBoost using RandomizedSearchCV')
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
params_sk = {
            "objective": "binary:hinge",
            'max_depth': 8,
           'n_estimators': 100, # the same as num_rounds in xgboost
           'subsample': 1.0,
           'colsample_bytree': 0.3,
           'seed': 23}

skrg = XGBRegressor(**params_sk)

skrg.fit(x_train[new_feat],y_train)

params_grid = {
   'learning_rate': st.uniform(0.1, 0.8),
   'max_depth': list(range(4, 20, 2)),
    'gamma': list(range(1, 15, 1)),
   'reg_alpha': list(range(1, 15, 1))}

search_sk = RandomizedSearchCV(skrg, params_grid, cv = 5, verbose=1)
search_sk.fit(x_train[new_feat], y_train)

# best parameters
print(search_sk.best_params_); print(search_sk.best_score_)

print('\n---------- FINAL MODEL ----------\n')
print('The final model is an ensemble of Random forest and extreme gradient boosting, the idea for this is explained in the write-up and even better in the notebook')
x = np.array(x_train)
x_xgb = np.array(x_train[new_feat])
x_xgb_test = np.array(x_test[new_feat])
y = np.array(y_train)

# after having run multiple times, I got around these params, so I'll leave them constant to have a more stable output
# {'gamma': 1, 'learning_rate': 0.4, 'max_depth': 18, 'reg_alpha': 12}

params = {"objective": "binary:hinge",
          "eta": 0.4,
          "max_depth": 18,
          "silent": 1,
          "nthread": -1,
          "seed": 24,
          "gamma": 1,
          "reg_alpha": 12
          }

num_trees = 400 

dtrain = xgb.DMatrix(x_xgb, y)
dvalid = xgb.DMatrix(x_xgb_test, y_test)
watchlist = [(dvalid, 'test'), (dtrain, 'train')]
print('Training XGBoost binary classifier (hinge) on all train (80%)')
gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=50, verbose_eval=False)

print('Saving xgb model')
filename = 'final_xgb.sav'
pickle.dump(gbm, open(filename, 'wb'))
filename = 'xgb_test.sav'
pickle.dump(x_xgb_test, open(filename, 'wb'))

RFC = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, max_features=best_max_features, n_jobs=-1, verbose=0)
print('Training Random Forest binary classifier on all train (80%)')
RFC.fit(x_train, y_train)

print('Saving xgb model')
filename = 'final_rf.sav'
pickle.dump(RFC, open(filename, 'wb'))
filename = 'rf_test.sav'
pickle.dump(x_test, open(filename, 'wb'))

filename = 'y_test.sav'
pickle.dump(y_test, open(filename, 'wb'))

print('Testing on test set (20%)')
y_hat_rf = RFC.predict(x_test)
y_hat_xgb = gbm.predict(xgb.DMatrix(x_xgb_test))


y_hat_ensemble = y_hat_rf * y_hat_xgb

accuracy = mean(y_hat_ensemble == y_test)
accuracy_rf = mean(y_hat_rf == y_test)
accuracy_xgb = mean(y_hat_xgb == y_test)

print('Accuracy XGBoost: '+str(accuracy_xgb))
print('Accuracy Random Forest: '+str(accuracy_rf))

print('\nFinal model (RF+XGB logical AND ensemble) accuracy : '+str(accuracy))

