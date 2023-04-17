import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from src.utils import save_object
import pickle
import warnings
warnings.filterwarnings('ignore')

class ModelTrainerConfig:
      trainset_train_file_path = os.path.join(os.getcwd(),'model_artifacts','train_test_dataset','training_trainset.csv')
      trainset_test_file_path  = os.path.join(os.getcwd(),'model_artifacts','train_test_dataset','training_testset.csv')

class ModelTrainerPipeline:
       def __init__(self):
            self.file_paths = ModelTrainerConfig()

       def initiate_model_trainer_pipeline(self):
             logging.info("Model Trainer Pipeline are initiated")

             try:
                 logging.info("Model Trainer Pipeline are initiated with list of the ML Models")
                 models:dict = {'RandomForestClassifier':RandomForestClassifier(),
                                'ExtraTreesClassifier':ExtraTreesClassifier(),
                                'GradientBoostingClassifier':GradientBoostingClassifier(),
                                'HistGradientBoostingClassifier':HistGradientBoostingClassifier(),
                                'LogisticRegression':LogisticRegression(),
                                'SVC':SVC(),
                                'KNeighborsClassifier':KNeighborsClassifier()
                                }

                 training_dataset = pd.read_csv(self.file_paths.trainset_train_file_path)

                 for keys,ML_model in models.items():
                       ML_model.fit(training_dataset.iloc[:,:-1],training_dataset.iloc[:,-1])
                       save_object(keys,ML_model)
                       logging.info("{} model trained and artifacts are stored in model_performance subfolder 'train_models'".format(keys))
                       print("{} model trained and artifacts are stored in model_performance subfolder 'train_models'".format(keys),'\n')

             except Exception as e:
                   raise CustomException(e,sys)
             

if __name__=="__main__":
      obj = ModelTrainerPipeline()
      obj.initiate_model_trainer_pipeline()