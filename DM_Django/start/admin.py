from django.contrib import admin

# Register your models here.

from .models import Data, DataTesting

admin.site.register(Data)
admin.site.register(DataTesting)