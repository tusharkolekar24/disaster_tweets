from src.logger import logging
from src.exception import CustomException
from src.component.data_cleaning import DataCleaning,bert_model
import sys
import pandas as pd
import os
from dataclasses import dataclass
@dataclass

class dataloadingConfig:
      raw_trainset_path = os.path.join(os.getcwd(),'original_set','train.csv')
      raw_testset_path = os.path.join(os.getcwd(),'original_set','test.csv')

class DataLoading:
      def __init__(self):
          self.data_loading_paths = dataloadingConfig()
          
      def initiate_data_ingestion(self):
          logging.info("Data Ingestion Process Started")

          try:
             trainset = pd.read_csv(self.data_loading_paths.raw_trainset_path)
             logging.info("Dataset is Available for training and Ready to Start")

             cleaning_text = DataCleaning()
             clean_trainset_with_stopwords, clean_trainset_without_stopwords = cleaning_text.get_clean_data(trainset['text'].values)
             clean_trainset = pd.DataFrame({'text':[sentence for sentence in clean_trainset_with_stopwords],
                                            'target':trainset['target']})
             
             logging.info("Text Data is cleaned & ready for prepearing further process")
             clean_trainset.to_csv(os.path.join(os.getcwd(),'model_artifacts','clean_dataset','clean_trainset.csv'),
                                   index=False)
             
             testset = pd.read_csv(self.data_loading_paths.raw_testset_path)
             logging.info("Dataset is Available for testing and Ready to Start")
             
             logging.info("Test dataset is cleaned & store in model artifacts with clean dataset folder")
             clean_testset_with_stopwords, clean_testset_without_stopwords = cleaning_text.get_clean_data(testset['text'].values)

             clean_testset = pd.DataFrame({
                                          'text':[sentence for sentence in clean_testset_with_stopwords],
                                         })
             
             clean_testset.to_csv(os.path.join(os.getcwd(),'model_artifacts','clean_dataset','clean_testset.csv'),
                                   index=False)
             logging.info("Test dataset is cleaned & store in model artifacts with clean dataset folder")

             logging.info("Sentence Embedding start with using BERT Models for train datasets")
             embedded_trainset_sentence = bert_model.encode(clean_trainset_with_stopwords)
 
             ready_embedded_trainset = pd.DataFrame(embedded_trainset_sentence,
                                                    columns=[f'feature_{count}' for count in range(embedded_trainset_sentence.shape[1])])

             ready_embedded_trainset['target'] = trainset['target'].values

             ready_embedded_trainset.to_csv(os.path.join(os.getcwd(),'model_artifacts','embedded_dataset','train.csv'),
                                            index=False)
             logging.info("Sentence Embedding Finished with using BERT Models for Training datasets")
             logging.info("Training datasets has shape of {}".format(ready_embedded_trainset.shape)) 

             logging.info("Sentence Embedding start with using BERT Models for Test datasets") 
             embedded_testset_sentence  = bert_model.encode(clean_testset_with_stopwords) 

             ready_embedded_testset = pd.DataFrame(embedded_testset_sentence,
                                                    columns=[f'feature_{count}' for count in range(embedded_testset_sentence.shape[1])])
             
             logging.info("Sentence Embedding Finished with using BERT Models for Testing datasets") 
             ready_embedded_testset.to_csv(os.path.join(os.getcwd(),'model_artifacts','embedded_dataset','test.csv'),
                                            index=False)
             logging.info("Testing datasets has shape of {}".format(ready_embedded_testset.shape)) 

          except Exception as e:
                 raise CustomException(e,sys)
          
if __name__=='__main__':
     object = DataLoading()
     object.initiate_data_ingestion()
