from django.shortcuts import render

# algoritm
import pandas as pd
import numpy as np
import re
import sklearn

from .models import Analisis
from start.models import DataTesting

# Create your views here.
def index(request):
    # query
    # posts = Analisis.objects.all()
    # posts = Data.objects.all()

    # context = {
    #     'data_sentimen' : posts,
        
    # }

    # importing necessary libraries
    
    testing = DataTesting.objects.all()
    
    df = pd.DataFrame(
            list(
                Analisis.objects.all().values(
                    "text","sentimen"
                )
            )
        )

    # df = pd.read_csv('static/dataset/preprocessing/sentimenReady2.csv')
    # df.head()
    # 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import GaussianNB

    vectorizer = TfidfVectorizer (max_features=3000)
    model_g = GaussianNB()

    v_data = vectorizer.fit_transform(df['text']).toarray()
    
    # print (v_data)


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(v_data, df['sentimen'], test_size=0.1, random_state=0)
    model_g.fit(X_train,y_train)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    fileTestResult =''
    if request.GET.get('text',None):
        Test = request.GET.get('text')
        df_T = pd.DataFrame([
            Test
        ])
        tf_custom = vectorizer.transform(df_T[0]).toarray()
        # 
        y_predsTest = model_g.predict(tf_custom)
    elif testing.exists():
        dataTest = pd.DataFrame(
                DataTesting.objects.all().values(
                    "testText"
                )
            )
        Test = ''
        tf_custom = vectorizer.transform(dataTest['testText']).toarray()
        y_predsTest = model_g.predict(tf_custom)
        y_predsTest = pd.DataFrame(y_predsTest, columns=['sentimen'])
        y_predsTest = dataTest.join(y_predsTest)
        fileTestResult = y_predsTest.to_dict(orient='records')
    else:
        tf_custom = '0'
        y_predsTest = ''
        Test= ''    
    y_preds = model_g.predict(X_test)
        



    # print(confusion_matrix(y_test,y_preds))
    # print(classification_report(y_test,y_preds))
    # print('nilai akurasinya adalah ',accuracy_score(y_test, y_preds))

    # print(y_test,X_test)
    # html_string = df.to_html(index=False)
    # mengubah data frame menjadi dictionary

    data_dict = df.to_dict(orient='records')
    context = {
        'data_sentimen' : data_dict,
        'matrix' : '11',
        'report' : '11',
        'akurasi' : '11',
        'matrix' : confusion_matrix(y_test,y_preds),
        'report' : classification_report(y_test,y_preds),
        'akurasi' : accuracy_score(y_test, y_preds),
        'textPredict' : Test,
        'sentimenPredict' : y_predsTest,
        'testingTable': fileTestResult,
    }
    
    return render(request, 'sentimen/index.html', context)