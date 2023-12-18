from django.shortcuts import render

# algoritm
import pandas as pd
import numpy as np
import re
import sklearn
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

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
    test = Analisis.objects.all()
    if test.exists():
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
        # positive = Analisis.objects.filter(sentimen='positive').count() # menghitung jumlah data sentimen positif
        # neutral = Analisis.objects.filter(sentimen='neutral').count() # menghitung jumlah data sentimen netral
        # negative = Analisis.objects.filter(sentimen='negative').count() # menghitung jumlah data sentimen negatif
        # sentimen = [positive, neutral, negative] # membuat list yang berisi jumlah data sentimen
        # label = ['Positive', 'Neutral', 'Negative'] # membuat list yang berisi label sentimen
        # plt.bar(label, sentimen) # membuat bar plot dari data sentimen dan label
        # plt.xlabel('Sentimen') # memberi label sumbu x
        # plt.ylabel('Jumlah Data') # memberi label sumbu y
        # plt.title('Visualisasi Data Sentimen') # memberi judul plot
        # plt.savefig('static/asset/visual/sentimen.png') # menyimpan plot sebagai file gambar dengan nama 'sentimen.png'
        # Visualsisasi Sentimen
        figSentimen = px.pie(df, names='sentimen', title = 'Sentimen', width=800, height=400)
        figSentimen.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#182B2D",
        )
        chart = figSentimen.to_html()
        data_dict = df.to_dict(orient='records')

        # visualisasi Word count
        countWord = Analisis.objects.values_list('text', flat=True)
        words = []
        for text in countWord:
            cleaned_text = re.sub(r'[^\w\s]', '', text)
            words.extend(cleaned_text.lower().split())
        word_count = {}
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        top_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:10]
        df_top_words = pd.DataFrame(top_words, columns=['kata', 'jumlah'])
        # data_word = px.Analisis.gapminder().query("country == 'Canada'")
        figWord = px.bar(df_top_words, x='kata', y='jumlah', title="10 Kata Teratas",)
        figWord.update_layout(
            xaxis_title="Kata",
            yaxis_title="Jumlah",
        )
        figWord.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#182B2D",
        )
        chartWord = figWord.to_html()

        y_preds_train = model_g.predict(X_train)

        # figTrain = px.histogram(x=y_preds_train, title="Distribusi Hasil Prediksi Data Training", labels={'x': 'Sentimen'})
        # figTrain.update_layout(
        #     xaxis_title="Sentimen",
        #     yaxis_title="Jumlah",
        # )
        # chartTrain = figTrain.to_html()
        
        # figTrainScatter = px.scatter(x=range(len(y_preds_train)), y=y_preds_train, title="Hasil Prediksi Data Training", labels={'x': 'Data', 'y': 'Sentimen'})
        # figTrainScatter.update_layout(
        #     xaxis_title="Data",
        #     yaxis_title="Sentimen",
        # )
        # chartTrain = figTrainScatter.to_html()

        y_preds = model_g.predict(X_test)
        cm = confusion_matrix(y_test, y_preds)

        figConfusionMatrix = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['True Negative', 'True Positive'],
            colorscale='Viridis',
            reversescale=True,
        ))

        figConfusionMatrix.update_layout(
            title='Confusion Matrix Heatmap',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
        )
        figConfusionMatrix.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#182B2D",
        )
        chartTrain = figConfusionMatrix.to_html()

        
        target_names = ['negative', 'positive']

        # target_names = ['Negative', 'Positive']
        classification_rep = classification_report(y_test, y_preds, target_names=target_names, output_dict=True)
        figClassificationReport = px.bar(
            x=target_names * 3,  # Repeat class names for each metric
            y=[classification_rep[class_name.lower()][metric] for class_name in target_names for metric in ['precision', 'recall', 'f1-score']],
            color=['Precision', 'Recall', 'F1-Score'] * len(target_names),
            barmode='group',
            labels={'y': 'Score', 'x': 'Class', 'color': 'Metric'},
            title='Classification Report Metrics by Class',
        )
        figClassificationReport.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#182B2D",
        )
        chartClassificationReport = figClassificationReport.to_html()

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
            'sentimenDataVisual': chart,
            'countWord':top_words,
            'wordDataVisual': chartWord,
            'trainDataVisual' : chartTrain,
            'reportDataVisual' : chartClassificationReport,
        }
    else:
        context= {
            'text' : 'belum memiliki data',
            'status' : 'none',
        }
        
    return render(request, 'sentimen/index.html', context)