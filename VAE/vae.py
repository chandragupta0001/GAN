import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, im_chan=3, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        self.enc = nn.Sequential(
            self.make_enc_block(im_chan, hidden_dim),
            self.make_enc_block(hidden_dim, hidden_dim * 2),
            self.make_enc_block(hidden_dim * 2, hidden_dim * 4),
            self.make_enc_block(hidden_dim * 4, hidden_dim * 8),
            self.make_enc_block(hidden_dim * 8, hidden_dim * 8),
            self.make_enc_block(hidden_dim * 8, 2 * output_chan, kernel_size=4, stride=1,padding=0 ,final_layer=True)
        )

    def make_enc_block(self, in_channels, out_channels, kernel_size=4, stride=2,padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding)
            )

    def forward(self, x):
        x = self.enc(x)
        print(x.shape)
        enc = x.view(len(x), -1)
        #         print(enc.shape)
        return enc[:, :self.z_dim], enc[:, self.z_dim:].exp()


class Decoder(nn.Module):
    def __init__(self, z_dim=32, im_chan=3, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),  # ch x 4 x4
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),  # ch x 8 x8
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),  # ch x 16 x 16
            self.make_gen_block(hidden_dim * 2, hidden_dim * 1, kernel_size=4, stride=2, padding=1),  # ch x 32 x 32
            self.make_gen_block(hidden_dim * 1, hidden_dim * 1, kernel_size=4, stride=2, padding=1),  # ch x 64 x 64
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True),  # 128 X 128
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Sigmoid(),
            )

    def forward(self, noise):

        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, z_dim=32, im_chan=3, hidden_dim=64):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(im_chan, z_dim)
        self.decoder = Decoder(z_dim, im_chan)

    def forward(self, images):
        mean, std = self.encoder(images)
        dist = Normal(mean, std)
        z = dist.rsample()
        decoding = self.decoder(z)

        return decoding, dist

reconstruction_loss=nn.MSELoss(reduction='sum')

from torch.distributions.kl import kl_divergence
def kl_divergence_loss(q_dist):
    return kl_divergence(
    q_dist,Normal(torch.zeros_like(q_dist.mean),torch.ones_like(q_dist.stddev))
    ).sum(-1)


from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

batch_size=64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize((128,128))
])

data=ImageFolder(root="C:\\Users\\chand\\Desktop\\MY_FACE",transform=transform)
dataloader=DataLoader(dataset=data,batch_size=batch_size,shuffle=True)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16, 8)

from torchvision.utils import make_grid
from tqdm import tqdm
import time


def show_tensor_images(image_tensor, num_images=4, size=(3, 128, 128)):
    print(image_tensor.shape)
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


device = 'cuda'
vae = VAE().to(device)
print(vae.parameters())
# model_parameters = filter(lambda p: p.requires_grad, vae.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)
vae_opt = torch.optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(10):
    for _,(images, _) in enumerate(dataloader):
        images = images.to(device)
        vae_opt.zero_grad()
        recon_images, encoding = vae(images)
        loss = reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding).sum()
        loss.backward()
        vae_opt.step()
    plt.subplot(1, 2, 1)
    show_tensor_images(images)
    plt.title("True")
    plt.subplot(1, 2, 2)
    show_tensor_images(recon_images)
    plt.title("Reconstructed")
    plt.show(block=False)
