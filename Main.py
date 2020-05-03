import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.externals import joblib
import pandas as pd
from google.cloud import storage

rng = 1
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))

print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor(objective='reg:linear',silent=False, nthread=-1,booster='gbtree',eval_metric='mae',random_state = rng)
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=0)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)
joblib.dump(clf, 'Boston_Model.joblib')

client = storage.Client(project='mlserverless')
bucket_name = "boston_model"
bucket = client.get_bucket(bucket_name)
blob = bucket.blob('Boston_Model.joblib')
blob.upload_from_filename('Boston_Model.joblib')

x0 = ['{:.2f}'.format(i) for i in X[0]]
x0 = pd.Series(x0)
with open("input.json", "w") as myfile:
    myfile.write("["+",".join(map(str, x0))+"]")
