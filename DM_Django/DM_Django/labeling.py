import nltk
import pandas as pd
import numpy as np
import re 

def labeling():
    
    # !pip install transformers
    # !pip install googletrans==3.1.0a0

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon")
    data = pd.read_csv("static/dataset/input/politikDInastiSelected.csv")
    data.head()


    from googletrans.client import Translator
    translator = Translator()

    from transformers import pipeline
    sentiment_classifier = pipeline('sentiment-analysis')

    def clean_tweet(tweet):
        return ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", tweet).split())

    data['label'] = data["full_text"].str.encode('ascii','ignore').apply(translator.translate, src='id', dest='en')
    data['label'] = data['label'].apply(getattr, args=('text',))
    data['label'] = data.apply(lambda x: clean_tweet(x["label"]), axis=1)

        # df = (
        #     data
        #     .assign(sentiment = lambda x: x['label'].apply(lambda s: sentiment_classifier(s)))
        #     .assign(
        #         label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
        #         score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
        #     )
        # )


    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["label"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["label"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["label"]]
    data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["label"]]
    data.head()


    score = data["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0 :
            sentiment.append('Positive')
        elif i < 0 :
            sentiment.append('Negative')
    data["label"] = sentiment
    data.head()

    # print(data["Sentiment"].value_counts())
    data.to_csv("static\dataset\labeling\labeling20.csv")
    return data



# def sentimen(request):
#     return render(request, 'sentimen.html')



# def index(request):
#     return HttpResponse("<h1 class='text-center'>hallo dunia</h1>")