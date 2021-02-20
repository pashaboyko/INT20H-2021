import contractions
import string
import re
import joblib
import logg
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('stopwords')
del_columns = ['book_title', 'book_desc', 'book_genre', 'book_authors', 'book_format']



class Model_Pipeline:
    def __init__(self,  file):

        #self.log = logg.get_class_log(self)
        self.pipeline = joblib.load(file)

    def pipelineData(self, data):
        return self.pipeline.transform(data)




def preprocess(data_real, colum_name):
    def remove_punctuation(sentence):
        unnecessary_dict = {}
        for symb in string.punctuation:
            unnecessary_dict[symb] = ' '
        unnecessary_dict['\x96'] = ' '
        unnecessary_dict['\x85'] = ' '
        unnecessary_dict['´'] = ' '
        unnecessary_dict['\x97'] = ' '
        unnecessary_dict['…'] = ' '
        unnecessary_dict['’'] = ' '
        unnecessary_dict['\x91'] = ' '

        s = sentence.replace('<br />', '')
        s = s.translate(s.maketrans(unnecessary_dict))
        return s

    def remove_stopwords(sentence):
        without_sw = []
        stop_words = stopwords.words('english')
        stop_words.remove('not')
        stop_words.remove('no')
        
        words = sentence.split()
        for word in words:
            if word not in stop_words:
                without_sw.append(word)

        res = ' '.join(without_sw)
        return res

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in text.split()]
        return " ".join(words)
    
    data = data_real.copy()
    #all to lowercase
    data[colum_name] = data[colum_name].str.lower()
    #change contractions (i've, don't ...) to full forms (i have, do not)
    data[colum_name] = data[colum_name].apply(lambda x: " ".join([contractions.fix(word) for word in str(x).split()]))
    #remove punctuation
    data[colum_name] = data[colum_name].apply(remove_punctuation)
    #remove numbers
    data[colum_name] = data[colum_name].apply(lambda s: re.sub('\d+', ' ', s))
    #remove stopwords
    data[colum_name] = data[colum_name].apply(remove_stopwords)
    #remove single letters
    data[colum_name] = data[colum_name].apply(lambda s: re.sub('\b[a-zA-Z]\b', ' ', s))
    #remove excess spaces
    data[colum_name] = data[colum_name].apply(lambda s: re.sub(' +', ' ', s))
    #lemmatize
    data[colum_name] = data[colum_name].apply(lemmatize_text)
    #tokenize
    data[f'tokenized_{colum_name}'] = data[colum_name].apply(lambda s: s.split())
    #add number of words column
    data[f'word_num_{colum_name}'] = data[f'tokenized_{colum_name}'].str.len()
    



class CleaningTextData(BaseEstimator, TransformerMixin):
    def __init__(self, del_columns=del_columns):
        self.del_columns = del_columns 
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        data_new = X.copy()
        
        # cleaning 
        data_new = preprocess(data_new, 'book_desc')
        data_new['book_genre_list'] = data_new['book_genre'].apply(lambda x: list(set((" ".join(str(x).split('|'))).split())) if x is not None else ['ok'])
        data_new['book_authors_list'] = data_new['book_authors'].apply(lambda x: list(set((" ".join(str(x).split('|'))).split())) if x is not None else ['ok'])
        data_new.book_format = data_new.book_format.fillna(data_new.book_format.mode())
        data_new = preprocess(data_new, 'book_format')
        data_new.book_pages = data_new['book_pages'].apply(lambda x: int(x.split()[0]) if len(str(x).split()) > 1 else np.nan)
        data_new = preprocess(data_new, 'book_title')
        data_new = data_new.drop(self.del_columns, axis=1)
        
        # to string some columns
        self.columns_str = ['tokenized_book_desc', 'book_genre_list', 'book_authors_list', 'tokenized_book_format', 'tokenized_book_title']
        
        for name in self.columns_str:
            data_new[name] = data_new[name].apply(lambda x: np.nan if len(x)==0 or x[0] == 'ok' else ' '.join(x))
            
        # come back to nan
        columns_back = ['book_image_url',
                         'tokenized_book_desc',
                         'word_num_book_desc',
                         'book_genre_list',
                         'book_authors_list',
                         'tokenized_book_format',
                         'tokenized_book_title']
        
        for name in columns_back:
            data_new[name] = data_new[name].apply(lambda x: np.nan if x == 'nan' else x)
            
        return data_new
    
class FillingNaN(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
#         self.columns_input = columns_input
        
    def fit(self, X, y=None):
        print('fit filling na')

        data = X.copy()
        
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(data)
        
        return self
    
    def transform(self, X):
        data = X.copy()
        data_subset_transformed = self.imputer.transform(data)
        data = pd.DataFrame(data_subset_transformed, columns=data.columns)
        
        return data

columns_idf = ['tokenized_book_desc', 'book_genre_list', 'book_authors_list', 'tokenized_book_format', 'tokenized_book_title']

class TfIdf(BaseEstimator, TransformerMixin):
    def __init__(self, columns_idf=columns_idf, max_features=10000):
        self.columns_idf = columns_idf
        self.max_features = max_features
        self.model_dic = {}

    def fit(self, X, y=None):
        print('fit tfidf')
        data = X.copy()
        
        for name in self.columns_idf:
            self.model_dic[name] = TfidfVectorizer(stop_words='english', max_features=self.max_features)
            self.model_dic[name].fit(data[name].values)
        return self
    
    def transform(self, X, y=None):
        data = X.copy()
        data_new = data.copy()
        
        for name, model in self.model_dic.items():
            print
            data_transformed = model.transform(data[name].values).toarray()
            data_transformed = pd.DataFrame(data_transformed, columns=[f'{name}_{x}' for x in range(data_transformed.shape[1])])
            data_new = pd.concat([data_new, data_transformed], axis=1)
        
        data_new = data_new.drop(list(self.model_dic.keys()) + ['book_image_url'], axis=1)
        
        return data_new
    
