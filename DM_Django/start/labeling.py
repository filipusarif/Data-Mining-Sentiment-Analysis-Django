import pandas as pd
import numpy as np
import re 
from .models import Data

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")

class Labeling:
    def proses(data):
        data = pd.DataFrame(
            list(
                data
            )
        )

        from googletrans.client import Translator
        translator = Translator()

        from transformers import pipeline
        sentiment_analyzer = pipeline("sentiment-analysis")

        def clean_tweet(tweet):
            return ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", tweet).split())
        
        data['text'] = data[1]
        data['text'] = data.apply(lambda x: clean_tweet(x["text"]), axis=1)
        data['sentimen'] = data['text'].str.encode('ascii','ignore').apply(translator.translate, src='id', dest='en')
        data['sentimen'] = data['sentimen'].apply(getattr, args=('text',))
        data['sentimen'] = data.apply(lambda x: clean_tweet(x["sentimen"]), axis=1)

        sentiments = SentimentIntensityAnalyzer()
        data['sentimen'] = [sentiments.polarity_scores(i)["compound"] for i in data['sentimen']]


        score = data["sentimen"].values
        sentiment = []
        for i in score:
            if i >= 0 :
                sentiment.append('Positive')
            elif i < 0 :
                sentiment.append('Negative')
        data[2] = sentiment
        
        return data.to_dict(orient='records')
        
