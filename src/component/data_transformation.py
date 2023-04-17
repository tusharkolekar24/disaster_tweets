import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass

class DataTransformationConfig:
      cleaned_train_dataset_path = os.path.join(os.getcwd(),'model_artifacts','embedded_dataset','train.csv')
      trainset_path = os.path.join(os.getcwd(),'model_artifacts','train_test_dataset','training_trainset.csv')
      testset_path  = os.path.join(os.getcwd(),'model_artifacts','train_test_dataset','training_testset.csv')

class DataTransformation:
      def __init__(self):
            self.trainset_filepath = DataTransformationConfig()

      def initialize_data_transformations(self):
            logging.info("Data Transformation Started")
            
            try:
                trainset = pd.read_csv(self.trainset_filepath.cleaned_train_dataset_path) 

                logging.info("Trainset successfully loaded & ready for Transformations")

                X_train , X_test = train_test_split(trainset,train_size=0.75,random_state=42)

                X_train.to_csv(self.trainset_filepath.trainset_path,
                               index=False)      

                X_test.to_csv(self.trainset_filepath.testset_path,
                               index=False) 
                
                logging.info("Trainset successfully divided into train-test split with 75-25%")

            except Exception as e:
                   raise CustomException(e,sys)      

if __name__=="__main__":
      obj = DataTransformation()
      obj.initialize_data_transformations()