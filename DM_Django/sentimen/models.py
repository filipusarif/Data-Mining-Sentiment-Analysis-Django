from django.db import models

# Create your models here.

class Analisis(models.Model):
    text = models.TextField()
    sentimen = models.CharField(max_length=8)

    def __str__(self):
        return "{}. {}".format(self.id ,self.text)