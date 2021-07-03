import torch.nn as nn
import torch
from torchsummary import summary
def weights_init_normal(m):
    classname=m.__class__.__name__
    if classname.find("Conv")!=-1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
        if hasattr(m,'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data,0)
    elif classname.find("BatchNorm2d")!=-1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data,0.0)


#ResNet

class ResidualBlock(nn.Module):
    def __init__(self,in_features):
        super(ResidualBlock, self).__init__()

        self.block=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features,in_features,3),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self,x):
        return x+self.block(x)


class GeneratorResnet(nn.Module):
    def __init__(self,input_shape,num_residual_blocks):
        super(GeneratorResnet, self).__init__()

        channels=input_shape[0]

        out_features=64
        model=[
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels,out_features,7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features=out_features

        #downsampling

        for _ in range(2):
            out_features *=2
            model+=[
                nn.Conv2d(in_features,out_features,3,stride=2,padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features=out_features

        #residual block

        for _ in range(num_residual_blocks):
            model+=[ResidualBlock(in_features)]

        #upsampling
        for _ in range(2):
            out_features //=2
            model+=[
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features,out_features,3,stride=1,padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features=out_features

        model+=[nn.ReflectionPad2d(channels),nn.Conv2d(out_features,channels,7),nn.Tanh()]

        self.model=nn.Sequential(*model)

    def forward(self,x):
        return self.model(x)




#Discriminator

class Discriminator(nn.Module):
    def __init__(self,input_shape):
        super(Discriminator,self).__init__()
        channels,height,width=input_shape
        
        self.output_shape=(1,int(height// 2**4),int(width// 2**4))
        
        def discriminator_block(in_filters,out_filters,normalize=True):
            layers=[nn.Conv2d(in_filters,out_filters,4,stride=2,padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model=nn.Sequential(
            *discriminator_block(channels,64,normalize=False),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(512,1,4,padding=1)
        )
    
    def forward(self,img):
        return self.model(img)


if __name__=='__main__':
    gen = GeneratorResnet([3, 256, 256], 2).to(device='cuda')
    summary(gen, (3, 256, 256))
    desc = Discriminator((3, 256, 256)).to(device='cuda', )
    summary(desc, (3, 256, 256))
