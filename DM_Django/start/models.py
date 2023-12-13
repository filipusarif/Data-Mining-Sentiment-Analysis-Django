from django.db import models

# Create your models here.

class Data(models.Model):
    text = models.TextField()
    sentimen = models.CharField(max_length=9)

    def __str__(self):
        return "{}. {}".format(self.id ,self.text)
    
    
class DataTesting(models.Model):
    testText = models.TextField()

    def __str__(self):
        return "{}. {}".format(self.id ,self.testText)