from src.logger import logging
from src.exception import CustomException
import pandas as pd

import sys 
import os
import pickle 

import warnings
warnings.filterwarnings('ignore')

ml_models = pickle.load(open(os.path.join('model_performance','train_models','HistGradientBoostingClassifier.pkl'),'rb'))
testdt = pd.read_csv(os.path.join('model_artifacts','embedded_dataset','test.csv'))



originaldt = pd.read_csv(os.path.join('original_set','test.csv'))
y_pred_validation = ml_models.predict(testdt.values)

submission = pd.DataFrame({"id":originaldt["id"].values,'target':y_pred_validation})
submission.to_csv(os.path.join(os.getcwd(),'submission','submission.csv'),index=False)