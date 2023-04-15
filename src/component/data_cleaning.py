from sentence_transformers import SentenceTransformer
import pandas as pd
import re
import unicodedata
import pandas as pd
from textblob import TextBlob
import numpy as np
import nltk
import json

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
class DataCleaning:
      def __init__(self):  
            pass
       
      def remove_accented_chars(self,text):    
            """
            Accent Normalization using following functions. 
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']
            """
            cleaning_sen = [unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8', 'ignore') 
                            for sentence in text]    
            return cleaning_sen
    
      def remove_html_elements(self,text):
            """
            Remove Unwanted Html tags or Element present in given sentences.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            cleaning_sen1 = [re.sub(r'<\S+>|<\S+ >' ,' ',sentence.replace("/",'')) for sentence in text]
            cleaning_sen  = [re.sub(r'\[.*?\]' ,' ',sentence) for sentence in cleaning_sen1]
            return cleaning_sen
        
      def remove_punctuation(self,text):
            """
            Unwanted punctuation present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            cleaning_sen = [''.join([char.lower() for char in sentence if char not in '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~']) for sentence in text]
            return cleaning_sen
    
      def remove_hashtags(self,text):
            """
            Unwanted hashtags present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            cleaning_sen = [re.sub('@\S+','',sentence).strip() for sentence in text]
            return cleaning_sen
        
      def remove_emojis(self,text):
            """
            Unwanted emojis present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            #cleaning_sen = [re.sub('#\S+','',sentence).strip() for sentence in text]
            cleaning_sen=[re.sub(r"["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", "", sentence)  for sentence in text]         
            return cleaning_sen
        
      def remove_digits(self,text):
            """
            Unwanted digits present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            cleaning_sen = [re.sub('\d+','',sentence).strip() for sentence in text]
            return cleaning_sen  

      def remove_url(self,text):
            """
            Unwanted URL present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            cleaning_sen = [re.sub('http://\S+|https://\S+','',sentence) for sentence in text]
            return cleaning_sen
        
      def space_removal(self,text):
            """
            Unwanted space present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """        
            cleaning_sen = [' '.join([word.replace("\n"," ") for word in sentence.split(" ") if word !='']) for sentence in text]
            return cleaning_sen
        
      def remove_contractions(self,text):
            """
            Unwanted space present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """          
            import contractions
            clean_sent = [contractions.fix(sentence) for sentence in text]
            return clean_sent 
        
      def autocorrect_sentence(self,text):
            """
            Unwanted space present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """ 
            import itertools
            from autocorrect import Speller    
            spell = Speller(lang='en')    
            clean_sent = [spell(''.join(''.join(s)[:2] for _, s in itertools.groupby(sentence))) for sentence in text]
            return clean_sent  

      def remove_stopwords(self,text):
            """
            Unwanted space present in given text will be softly handle by same functions.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            import nltk

            #Stop words present in the library
            stopwords = nltk.corpus.stopwords.words('english')
            
#             measurable = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS','RB','RBR','RBS','WRB']
#             filter_text = []
#             for sentence in text:
#                 sample_text = TextBlob(sentence).tags
#                 common_details = []
#                 for (word,tags) in sample_text:
#                      if tags in measurable:
#                         common_details.append(word)
#                 if len(common_details)!=0:
#                    filter_text.append(" ".join(common_details)) 

#                 else:
#                      filter_text.append(" ")   
                        
            #stopwords = json.load(open(r"D:\kaggel_comp\nlp-getting-started (2)\mapping.json",'r'))
            
            cleaning_sen = [' '.join([word for word in sentence.split(" ") if word not in stopwords]) 
                            for sentence in text]

            return cleaning_sen
        
      def lemmatizer_sentence(self,text):
            """
            It stems the word but makes sure that it does not lose its meaning.  
            Lemmatization has a pre-defined dictionary that stores the context of words and checks 
            the word in the dictionary while diminishing.

            The difference between Stemming and Lemmatization can be understood with the example provided below.

            Original Word	After Stemming	After Lemmatization
            goose	goos	goose
            geese	gees	goose
            """

            from nltk.stem import WordNetLemmatizer
            # nltk.download('wordnet')
            # nltk.download('omw-1.4')

            #defining the object for Lemmatization
            wordnet_lemmatizer = WordNetLemmatizer()
            
            lemm_text = [' '.join([wordnet_lemmatizer.lemmatize(word) for word in sentence.split(" ")]) for sentence in text]
           
            return lemm_text
        
      def stemming_sentence(self,text):
            """
            It is also known as the text standardization step where the words are stemmed or diminished 
            to their root/base form.  For example, words like ‘programmer’, ‘programming, ‘program’
            will be stemmed to ‘program’.
            But the disadvantage of stemming is that it stems the words such that its root form loses
            the meaning or it is not diminished to a proper English word. We will see this in the steps done below.
            """    
            #importing the Stemming function from nltk library
            from nltk.stem.porter import PorterStemmer

            #defining the object for stemming
            porter_stemmer = PorterStemmer()

            #defining a function for stemming
            stem_text = [' '.join([porter_stemmer.stem(word) for word in sentence.split(" ")]) for sentence in text]

            return stem_text
        
      def get_clean_data(self,text):
          """
          Data Cleaning will be taken by using various in build functions.
          Used Dataframe with column name and assigne value to text variable.

          Example :
                   text = spam_detection['messages']
          """
            
          updated_text = self.remove_accented_chars(text)
          updated_text = self.remove_url(updated_text)
          updated_text = self.remove_html_elements(updated_text)
          updated_text = self.remove_emojis(updated_text)
          updated_text = self.space_removal(updated_text)
          updated_text = self.remove_digits(updated_text)
          updated_text = self.remove_punctuation(updated_text)
          updated_text = self.remove_contractions(updated_text)
          updated_text = self.space_removal(updated_text)
          with_stopwords = self.lemmatizer_sentence(updated_text).copy()
          without_stopwords = self.remove_stopwords(with_stopwords)
          return with_stopwords,without_stopwords
      
if __name__=="__main__":
     pass