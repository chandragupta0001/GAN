import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from scipy.stats import truncnorm

def show_tensor_images(image_tensor,num_image=16,size=(3,64,64),nrow=3):
    image_tensor=(image_tensor+1)/2
    image_unflat=image_tensor.detach().cpu().clamp_(0,1)
    image_grid=make_grid(image_unflat[:num_image],nrow=nrow,padding=0)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.axis('off')
    plt.show()

def get_truncated_noise(n_samples,z_dim,truncation):
    truncated_noise=truncnorm.rvs(-truncation,truncation,size=(n_samples,z_dim))
    return torch.Tensor(truncated_noise)


class MappingLayers(nn.Module):

    def __init__(self,z_dim,hidden_dim,w_dim):
        super().__init__()
        self.mapping=nn.Sequential(
            nn.Linear(in_features=z_dim,out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )
    def forward(self,noise):
        return self.mapping(noise)


class InjectNoise(nn.Module):
    def __init__(self,channels):
        super(InjectNoise, self).__init__()
        self.weight=nn.Parameter(
            torch.randn((1,channels,1,1))
        )

    def forward(self,image):
        "image format n_samples, channels,width,height"
        noise_shape=(image.shape[0],1,image.shape[2],image.shape[3])
        noise=torch.randn(noise_shape,device=image.device)

        return image + noise*self.weight


class AdaIN(nn.Module):
    def __init__(self,channels,w_dim):
        super(AdaIN, self).__init__()
        self.instNorm=nn.InstanceNorm2d(channels)
        self.ys=nn.Linear(w_dim,channels)
        self.yb=nn.Linear(w_dim,channels)

    def forward(self, image,w):
        normalized_image=self.instNorm(image)
        style_scale=self.ys(w)[:,:,None,None]
        style_shift=self.yb(w)[:,:,None,None]

        transformed_image=style_scale*normalized_image + style_shift
        return transformed_image


class MicroStyleGANGeneratorBlock(nn.Module):
    def __init__(self,in_chan,out_chan,w_dim,kernel_size,startin_size,use_upsampling=True):
        super(MicroStyleGANGeneratorBlock, self).__init__()
        self.use_upsampling=use_upsampling

        if self.use_upsampling:
            self.upsample=nn.Upsample((startin_size,startin_size),mode='bilinear')
        self.conv=nn.Conv2d(in_chan,out_chan,kernel_size,padding=1)
        self.inject_noise=InjectNoise(channels=out_chan)
        self.adain=AdaIN(channels=out_chan,w_dim=w_dim)
        self.activation=nn.LeakyReLU(0.2)

    def forward(self,x,w):

        if self.use_upsampling:
            x=self.upsample(x)
        x=self.conv(x)
        x=self.inject_noise(x)
        x=self.adain(x,w)
        x=self.activation(x)
        return x
