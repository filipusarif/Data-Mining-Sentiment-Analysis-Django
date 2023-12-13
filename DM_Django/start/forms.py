from django import forms
# from .models import Files

class DocumentForm(forms.ModelForm):
    csv_file = forms.FileField(label='Pilih file CSV')