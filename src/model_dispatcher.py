from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import xgboost as xgb


"""
This file contains the models to be dispatched
to the training script.   
"""

models = {
    "log_reg": linear_model.LogisticRegression(),
    "random_forest": ensemble.RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=6),
    "XGboost": xgb.XGBRegressor(max_depth=5,
                              n_estimators=300,
                              use_label_encoder=False
                              )
}