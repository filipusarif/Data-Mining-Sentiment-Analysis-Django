import pandas as pd
import numpy as np
import re 
from .models import Data

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
# data_path = "D:\Github\Sentiment Analysis-DMProject-Django\dataset\politikDInastiSelected.csv"
# data = pd.read_csv(data_path)
# data.head()import pandas as pd
class Labeling:
    def proses(data):
        data = pd.DataFrame(
            list(
                data
            )
        )

        # from googletrans.client import Translator
        # translator = Translator()

        # from transformers import pipeline
        # sentiment_classifier = pipeline('sentiment-analysis')

        # def clean_tweet(tweet):
        #     return ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", tweet).split())

        # data['label'] = data[1].str.encode('ascii','ignore').apply(translator.translate, src='id', dest='en')
        # data['label'] = data['label'].apply(getattr, args=('text',))
        # data['label'] = data.apply(lambda x: clean_tweet(x["label"]), axis=1)

        # df = (
        #     data
        #     .assign(sentiment = lambda x: x['label'].apply(lambda s: sentiment_classifier(s)))
        #     .assign(
        #         label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
        #         score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
        #     )
        # )


        sentiments = SentimentIntensityAnalyzer()
        # data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["label"]]
        # data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["label"]]
        # data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["label"]]
        data['sentimen'] = [sentiments.polarity_scores(i)["compound"] for i in data[1]]
        data.head()


        score = data["sentimen"].values
        sentiment = []
        for i in score:
            if i >= 0 :
                sentiment.append('Positive')
            elif i < 0 :
                sentiment.append('Negative')
        data[2] = sentiment
        data.head()
        # return data
        return data.to_dict(orient='records')
        # return data.to_dict(orient='records').value('text','sentimen')
        # print(data["sentimen"].value_counts())

        # data.to_csv("sentimenAnalisis\dataset\labeling10.csv")