
from django.http import JsonResponse
from .serializer import ImageSerializer
from rest_framework.decorators import api_view
from .models import Image
import torch
import torchvision.transforms as transforms
from PIL import Image as pilimg
import torch.nn as nn
from .Tumor_model import Brain_Tumor

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
inc=3
num_classes=2
batch_size=50
parameters_path='backend/final.pth'
load=False

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150,150)),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

def predict(x):
        print('in predict')
        if str(type(x))=="<class 'PIL.JpegImagePlugin.JpegImageFile'>":
            tens=transform(x)
        else:
            tens=transform(pilimg.open(x))
        print(tens.shape)
        tens=tens.reshape([1,3,150,150]).to(device)
        model=Brain_Tumor(inc,num_classes)
        model=model.to(device)
        if load:
            loader=torch.load(parameters_path,map_location=device)
            model.load_state_dict(loader['model'])
            print('parameters loaded')
        pred=model(tens)
        sf=nn.Softmax(dim=1)(pred)
        a=torch.argmax(sf,dim=1)
        return a.item()

@api_view(['POST'])
def post(request):
    serializer=ImageSerializer(data=request.data)
    if serializer.is_valid():
        print('saved')
        serializer.save()
    img=Image.objects.all().values()[0]['image']
    print(img)
    result=predict(img)
    if result==1:result='POSITIVE (tumor detected)'
    else:result='NEGITIVE (no tumor detected)'
    Image.objects.all().delete()
    print('entry deleted')

    
    return JsonResponse({'output':result,'success':True})


