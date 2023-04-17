import pandas as pd
from textblob import TextBlob
import numpy as np
import nltk
import string

class FeatureEngineering:
      def __init__(self,raw_data):
            self.__additional_feature = pd.DataFrame(columns=[
                                                              #'punctuation_count',
                                                             'character_count',
                                                             'word_counts','word_density',
                                                              #'polarity','subjectivity',
                                                              'noun_count','pron_count',
                                                             'verb_count','adj_count','adv_count','other_count'])
            self.__raw_data = raw_data

      def punctuation_count(self,text):
            """
            Count existing Punctuations in the given text and summerized it as additional features.
            raw dataset with columns name must be parse as input to extract or generated punctuation counts.
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            import string
            common_details = []
            for sentence in text:
                symbols_list = [symbols for symbols in sentence if symbols in string.punctuation]
                if len(symbols_list)!=0:
                    common_details.append(len(symbols_list))    
                else:
                     common_details.append(0)
                        
            scaled_punct_counts = [np.log10(counts+1) for counts in common_details]
            return scaled_punct_counts 
        
      def get_hashtags(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            separate_links = []
            for sentence in text:
                strings = re.findall('@\S+',sentence.replace(":",'').replace(".",''))
                if len(strings)!=0:
                    common_details = []
                    for words_com in strings:
                        common_details.append(words_com.replace("@",'').lower())
                    separate_links.append(common_details)
                else:
                     separate_links.append([''])
            return separate_links  
        
      def get_emojis(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            separate_links = []
            for sentence in text:
                strings = re.findall('#\S+',sentence.replace(":",'').replace(".",''))
                if len(strings)!=0:
                    common_details = []
                    for words_com in strings:
                        if '/' not in words_com: 
                            common_details.append(words_com.replace("#",'').lower())
                        if '/' in words_com:
                             #print(words_com)
                             for sep_slash in words_com.split("/"):
                                 common_details.append(sep_slash.replace("#",'').lower())   

                    separate_links.append(common_details)
                else:
                     separate_links.append([''])
            return separate_links
        
      def character_count(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            length_sentence =[len(sentence) for sentence in text] 
            scaled_length_sent = [np.log10(count+1) for count in length_sentence]
            return scaled_length_sent

      def word_counts(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            word_counts_list = [len(sentence.split(" ")) for sentence in text]
            scaled_word_counts = [np.log10(count+1) for count in word_counts_list]
            return scaled_word_counts


      def word_density(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            import numpy as np
            length_sentence =[len(sentence)+1 for sentence in text]  
            word_counts_list = [len(sentence.split(" "))+1 for sentence in text]
            word_density_list = np.array(word_counts_list)/np.array(length_sentence)

            return word_density_list

      def get_polarity(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            polarity_details = []
            for sentence in text: # remove_punctuation
                textblob = TextBlob(str(sentence.encode('utf-8')))
                pol = textblob.sentiment.polarity
                polarity_details.append(pol)
            return polarity_details


      def get_subjectivity(self,text):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            subjectivity_details = []
            for sentence in text:#remove_punctuation
                textblob = TextBlob(str(sentence.encode('utf-8')))

                subj = textblob.sentiment.subjectivity
                subjectivity_details.append(subj)
            return subjectivity_details
        
      def pos_check(self,text,flag):
            """
            Used Dataframe with column name and assigne value to text variable.

            Example :
                     text = spam_detection['messages']            
            """
            # lets create a Part of speech Dictionary
            pos_dic = {
                'noun' : ['NN','NNS','NNP','NNPS'],
                'pron' : ['PRP','PRP$','WP','WP$'],
                'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
                'adj' :  ['JJ','JJR','JJS'],
                'adv' :  ['RB','RBR','RBS','WRB'],
                'other':  ['IN', 'CD', 'POS', 'RP', 'FW', 'MD', 'DT', 'CC', 'PDT', 'WDT']
                }

            pos_check_list =[]
            for sentence in text:
                sample_text = TextBlob(sentence).tags
                tags_counts = 1
                for (words,tags) in sample_text:
                    if tags in pos_dic.get(flag):
                        tags_counts+=1
                pos_check_list.append(np.log10(tags_counts))  

            return pos_check_list    

      def get_extracted_feature(self,text):
#          self.__additional_feature['punctuation_count'] = self.punctuation_count(self.__raw_data)
#           self.__additional_feature['polarity']          = self.get_polarity(text)
#           self.__additional_feature['subjectivity']      = self.get_subjectivity(text)
          self.__additional_feature['character_count']   = self.character_count(text)
          self.__additional_feature['word_counts']       = self.word_counts(text)
          self.__additional_feature['word_density']      = self.word_density(text)
          self.__additional_feature['noun_count']        = self.pos_check(text,'noun')
          self.__additional_feature['pron_count']        = self.pos_check(text,'pron')
          self.__additional_feature['verb_count']        = self.pos_check(text,'verb')
          self.__additional_feature['adj_count']         = self.pos_check(text,'adj')
          self.__additional_feature['adv_count']         = self.pos_check(text,'adv')
          self.__additional_feature['other_count']       = self.pos_check(text,'other')
          return self.__additional_feature