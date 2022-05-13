import os
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as tt
import torch
import torch.nn as nn
#import cv2
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from Telegram_BOT import send_image

DATA_DIR = 'dataset'
LABELS = 'file_list.txt'
BATCH_SIZE = 128
DEVICE = 'cuda'
EPOCHS = 100
SAMPLE_DIR = 'generated'

### Data functions ###

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def save_checkpoint(NeuralNet, folder="checkpoints", filename="model.pkl"):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    torch.save(obj={"state_dict": NeuralNet.state_dict()}, f=filepath)

def load_checkpoint(NeuralNet, folder="checkpoints", filename="model.pkl"):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError("No model in path {}".format(filepath))
    map_location = DEVICE
    checkpoint = torch.load(filepath, map_location=map_location)
    NeuralNet.load_state_dict(checkpoint["state_dict"], strict=False)


def save_samples(index):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    img_labels = open(LABELS, 'r').read().splitlines()

    to_upscale_path = os.path.join('data/64', img_labels[1])
    to_upscale_img = tt.ToTensor()(Image.open(to_upscale_path).convert('YCbCr')).to(DEVICE)

    to_upscale_img_ = tt.ToTensor()(Image.open(to_upscale_path).convert('YCbCr').resize((250, 250), Image.Resampling.BICUBIC))

    upscaled_path = os.path.join('data/250', img_labels[1])
    upscaled_img = tt.ToTensor()(Image.open(upscaled_path).convert('YCbCr'))

    netU.eval()
    corrected = to_upscale_img_.clone()
    corrected[0, :, :] = netU(to_upscale_img[None, None, 0, :, :])[0, 0, :, :].to('cpu')
    netU.train()
    images = torch.cat((to_upscale_img_, corrected, upscaled_img), dim=2)
    images = tt.ToPILImage('YCbCr')(images).convert('RGB')

    name = 'upscaled-images-{0:0=4d}.png'.format(index)
    images.save(os.path.join(SAMPLE_DIR, name))
    print('Saving', name)

### Data classes ###

class ImageDataset(Dataset):
    def __init__(self, file_list, to_upscale_dir, upscaled_dir, transform=None, target_transform=None):
        self.img_labels = open(file_list, 'r').read().splitlines()
        self.to_upscale_dir = to_upscale_dir
        self.upscaled_dir = upscaled_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        to_upscale_path = os.path.join(self.to_upscale_dir, self.img_labels[idx])
        to_upscale_img = Image.open(to_upscale_path).convert('YCbCr')
        upscaled_path = os.path.join(self.upscaled_dir, self.img_labels[idx])
        upscaled_img = Image.open(upscaled_path).convert('YCbCr')
        if self.transform:
            to_upscale_img = self.transform(to_upscale_img)[0, :, :].unsqueeze(0)
            #print(to_upscale_img.shape)
        if self.target_transform:
            upscaled_img = self.target_transform(upscaled_img)[0, :, :].unsqueeze(0)
            #print(upscaled_img.shape)
        return to_upscale_img, upscaled_img

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b1, b2 in self.dl:
            yield to_device(b1, self.device), to_device(b2, self.device)

    def __len__(self):
        return len(self.dl)

ds = ImageDataset(LABELS, 'data/64', 'data/250', transform=tt.ToTensor(), target_transform=tt.ToTensor())
dl = DeviceDataLoader(DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=6), DEVICE)


nz = 3
ngf = 64
nc = 1
ndf = 92
beta1 = 0.5
lr = 2e-4


class Upscaler(nn.Module):
    def __init__(self):
        super(Upscaler, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf // 4, ndf, kernel_size = 4, stride = 1, padding = 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.PixelShuffle(2),

            nn.Conv2d(ndf // 4, nc, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
    def forward(self, input):
        return self.main(input)

netU = Upscaler().to(DEVICE)
#load_checkpoint(netU, filename="upscaler.pkl")
criterion = nn.L1Loss()
optimizerU = torch.optim.Adam(netU.parameters(), lr=lr, betas=(0.5, 0.999))

def train_upscaler(to_upscale_img, upscaled_img):
    netU.zero_grad()
    output = netU(to_upscale_img)
    error = criterion(output, upscaled_img)
    error.backward()
    optimizerU.step()
    return error.mean().item()

for epoch in range(EPOCHS):
    # For each batch in the dataloader
    for to_upscale_img, upscaled_img in tqdm(dl):

        loss = train_upscaler(to_upscale_img, upscaled_img)

    print("Epoch [{}/{}]: loss: {:.4f}".format(epoch + 1, EPOCHS, loss))
    save_checkpoint(netU, filename = "upscaler.pkl")
    save_samples(epoch+1)
    send_image(fake_fname = 'upscaled-images-{0:0=4d}.png'.format(index), msg = "Epoch [{}/{}], loss: {:.4f}".format(epoch+1, EPOCHS, loss))

#save_samples(1)
