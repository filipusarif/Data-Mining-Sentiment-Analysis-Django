# myapp/views.py
from django.shortcuts import render, redirect
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import Data, DataTesting
from sentimen.models import Analisis
from .preprocessing import Preprocessing
from .labeling import Labeling
import csv

from django.http import JsonResponse
from django.core import serializers 

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

def index(request):
    if request.method == 'POST' and request.FILES.get('file_upload',None):
        Data.objects.all().delete()
        csv_file = request.FILES['file_upload']
        decoded_file = csv_file.read().decode('utf-8').splitlines()
        csv_reader = csv.reader(decoded_file)
        next(csv_reader, None)

        label = request.POST.get('label')
        if label == '1':
            # return redirect('/Sentimen/')
            csv_reader = Labeling.proses(csv_reader)

        for row in csv_reader:
            # Sesuaikan dengan struktur CSV Anda dan model Anda
            Data.objects.create(
                text=row[1],
                sentimen=row[2],
            )
            
        return redirect('/')  # Ganti 'success' dengan nama rute untuk halaman sukses
    
    if request.method == 'POST' and request.POST.get('textTest', None):
        DataTesting.objects.all().delete()
        data = request.POST.get('textTest')
        preprocessing()
        return redirect(f'/Sentimen/?text={data}')
        # if request.POST['textTest'] == '1':
        #     analisis()
        #     return redirect(f'/Sentimen/?text={data}')
        # else:
        #     preprocessing()
        #     return redirect(f'/Sentimen/?text={data}')
    
    if request.method == 'POST' and request.FILES.get('testCSV',None):
        DataTesting.objects.all().delete()
        test_CSV = request.FILES['testCSV']
        decoded_test = test_CSV.read().decode('utf-8').splitlines()
        test_Reader = csv.reader(decoded_test)
        next(test_Reader, None)

        for row in test_Reader:
            # Sesuaikan dengan struktur CSV Anda dan model Anda
            DataTesting.objects.create(
                testText=row[1],
            )

        return redirect('/Sentimen/')  # Ganti 'success' dengan nama rute untuk halaman sukses

        

    data = Data.objects.all()
    jumlah = Data.objects.count()

    context = {
        'data' : data,
        'jumlah': jumlah,
    }

    return render(request, 'start/index.html', context)

def analisis():
    Analisis.objects.all().delete()
    data = Data.objects.values('text', 'sentimen')
    Analisis.objects.bulk_create([Analisis(**d) for d in data])

def preprocessing():
    Analisis.objects.all().delete()
    data = Preprocessing.proses()
    Analisis.objects.bulk_create([Analisis(**d) for d in data])

def labeling(request):
    Data.objects.all().delete()
    Analisis.objects.all().delete()
    data = Labeling.proses()
    Data.objects.bulk_create([Data(**d) for d in data])
    data1 = Preprocessing.proses()
    Analisis.objects.bulk_create([Analisis(**d) for d in data1])
    return redirect('/Sentimen/')
    
