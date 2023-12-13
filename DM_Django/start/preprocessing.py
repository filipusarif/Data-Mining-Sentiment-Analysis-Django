import pandas as pd 
import numpy as np
import nltk
import string 
import re #regex library
from .models import Data


# Tokenize
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
nltk.download('punkt')
# Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')

# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# --------------------------- cleaning
def remove_tweet_special(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# NLTK word rokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords = stopwords.words('indonesian')

# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                    'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                    'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                    'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                    'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                    'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                    '&amp', 'yah'])
# convert list to dictionary
list_stopwords = set(list_stopwords)

term_dict = {}
normalizad_word_dict = {}
def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

# remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]
    

class Preprocessing():
    def proses():
        TWEET_DATA = pd.DataFrame(
            list(
                Data.objects.all().values(
                    "text","sentimen"
                )
            )
        )
        # TWEET_DATA = pd.read_csv("D:\Github\Sentiment Analysis-DMProject-Django\sentimenAnalisis\dataset\Tes1.csv")
        # TWEET_DATA.head()
        

        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_tweet_special)
        TWEET_DATA['text'] = TWEET_DATA['text'].str.lower()

        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_number)
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_punctuation)
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_whitespace_LT)
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_whitespace_multiple)
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(remove_singl_char)
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(word_tokenize_wrapper)

        normalizad_word = pd.read_excel("static\dataset\input\DSnormalisasi.xlsx")
        for index, row in normalizad_word.iterrows():
            if row[0] not in normalizad_word_dict:
                normalizad_word_dict[row[0]] = row[1] 
        
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(normalized_term)
        TWEET_DATA['text'].head(10)



        # for document in TWEET_DATA['text']:
        #     for term in document:
        #         if term not in term_dict:
        #             term_dict[term] = ' '
                    
        # print(len(term_dict))
        # print("------------------------")

        # for term in term_dict:
        #     term_dict[term] = stemmed_wrapper(term)
        #     print(term,":" ,term_dict[term])
            
        # print(term_dict)
        # print("------------------------")
        
        # TWEET_DATA['text'] = TWEET_DATA['text'].swifter.apply(get_stemmed_term)
        # print(TWEET_DATA['text'])
        
        
        TWEET_DATA['text'] = TWEET_DATA['text'].apply(stopwords_removal) 

        # ubah data frame ke numpy array
        return TWEET_DATA.to_dict(orient='records')
        # mengubah data frame menjadi dictionary
        # data_dict = df.to_dict(orient='records')

        # Analisis.objects.all().delete()
        # data = Data.objects.values('text', 'sentimen')
        # Analisis.objects.bulk_create([Analisis(**d) for d in data])
        # print(TWEET_DATA['text'].head())

    

    
    
    
    
    


# -------------------------------------- cleaning --------------------------------------


                


# -------------------------------------- Case Folding --------------------------------------



# -------------------------------------- Tokenizing --------------------------------------



















# -------------------------------------- Normalized and Stemming --------------------------------------








# -------------------------------------- stemm --------------------------------------


















# -------------------------------------- stopwords --------------------------------------



# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
# txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

# # convert stopword string to list & append additional stopword
# list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# ---------------------------------------------------------------------------------------



