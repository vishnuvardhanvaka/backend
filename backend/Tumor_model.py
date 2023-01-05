import torch
import torch.nn as nn                

def conv_block(in_channels,out_channels,pool=False):
    layers=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
    if pool : layers.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*layers)
#model
class Brain_Tumor(nn.Module):
    def __init__(self,inc,num_classes):

        super().__init__()
        self.conv=nn.Sequential(
          nn.Conv2d(3,32,kernel_size=3,padding=1),
          nn.ReLU(),
          
          nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

          nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
          nn.ReLU(),
          nn.MaxPool2d(2,2),

        )
        self.classifier=nn.Sequential(
          nn.Flatten(),
          nn.Linear(512*18*18,1024),
          nn.ReLU(),
          nn.Linear(1024,512),
          nn.ReLU(),
          nn.Linear(512,num_classes),
        )
        
        self.conv1=conv_block(inc,64)
        self.conv2=conv_block(64,128,pool=True)
        self.res1=nn.Sequential(
            conv_block(128,128),
            conv_block(128,128),
            )
        self.conv3=conv_block(128,256,pool=True)
        self.conv4=conv_block(256,512,pool=True)
        self.res2=nn.Sequential(
            conv_block(512,512),
            conv_block(512,512),
            
            )
        
        
        self.layer=nn.Sequential(
            
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512*18*18,num_classes),
        )
    def forward(self,x):
        
        x=self.conv(x)
        
        x=self.classifier(x)
       
        
        return x

    

















