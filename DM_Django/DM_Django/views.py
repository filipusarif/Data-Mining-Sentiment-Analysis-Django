from django.http import HttpResponse
from django.shortcuts import render, redirect

# from .forms import DocumentForm
# # import nltk
# # import pandas as pd
# # import numpy as np
# # import re 



# def index(request):

#     if request.method == 'POST':
#         print('ini adalah post')
#         print(request.POST['file_upload'])
#     else:
#         print('ini adalah get')
#     # df = labeling()
#     # data_dict = df.to_dict(orient='records')
#     # context = {
#     #     'data':data_dict,
#     #     'jumlah': df.shape[0],
#     # }
#     context = {
#         'data': 'hello',
#         'jumlah': '1122',
#     }
#     return render(request, 'index.html', context)


# def model_form_upload(request):
#     if request.method == 'POST':
#         form = DocumentForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             return redirect('sentimen')
#     else:
#         form = DocumentForm()
#     return render(request, 'index.html', {
#         'form': form
#     })