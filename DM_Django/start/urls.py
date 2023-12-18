from django.urls import include, path

from . import views

urlpatterns = [
    path('', views.index),
    path('labeling/', views.labeling, name='labeling'),
    path('analisis/', views.analisis, name='analisis'),
    path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('delete-data/', views.delete, name='delete'),
    # path("__reload__/", include("django_browser_reload.urls")),
]