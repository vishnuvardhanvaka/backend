from django.db import models

class Image(models.Model):
    image=models.ImageField(upload_to='media/images',default='',blank=False,null=False)
    time=models.DateTimeField(auto_now_add=True)
    


