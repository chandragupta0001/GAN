import os
import sys

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from models import *
from datasets import *
if __name__ == '__main__':
    os.makedirs("gen_images\%s" % "photo2monet", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.Tensor
    input_shape = (3, 256, 256)
    G_BA = GeneratorResnet(input_shape, 9).to(device)
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % ("monet2photo", 1)))

    transforms_ = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ]
    dataloader = DataLoader(
        ImageDataset("P:\\dataset\\monet2photo", transforms_,mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=4

    )
    print(dataloader.__len__())
    for i, batch, in enumerate(dataloader):
        images = batch["B"].type(Tensor)
        G_BA.eval()
        gen_image = G_BA(images)
        gen_image = gen_image.squeeze(dim=0)
        save_image(gen_image, "gen_images/%s/%s.jpg" % ("photo2monet", i), normalize=True)
        sys.stdout.write("\rdone %d/%d"%(i, dataloader.__len__()))


