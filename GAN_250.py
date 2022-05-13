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
from torchsummary import summary
from Telegram_BOT import send_image

DATA_DIR = 'data/250'
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


def save_samples(index, latent_tensors):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    fake_images = netG(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(SAMPLE_DIR, fake_fname), nrow=4)
    print('Saving', fake_fname)

### Data classes ###

class ImageDataset(Dataset):
    def __init__(self, file_list, img_dir, transform=None, target_transform=None):
        self.img_labels = open(file_list, 'r').read().splitlines()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b, _ in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

ds = ImageDataset(LABELS, DATA_DIR, transform=tt.Compose([tt.Grayscale(1), tt.ToTensor()]))
dl = DeviceDataLoader(DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=6), DEVICE)

nz = 100
ngf = 85
nc = 1
ndf = 40
beta1 = 0.5
lr = 1e-3

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 63 x 63
            nn.ConvTranspose2d( ngf, ngf, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 125 x 125
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 250 x 250
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 250 x 250
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 125 x 125
            nn.Conv2d(ndf, ndf * 2, 3, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG = Generator().to(DEVICE)
netG.apply(weights_init)
netD = Discriminator().to(DEVICE)
netD.apply(weights_init)

summary(netG, (100, 1, 1))
summary(netD, (1, 250, 250))

criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(16, nz, 1, 1, device=DEVICE)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

iters = 0

def train_discriminator(real_images):
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        b_size = real_images.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)
        # Forward pass real batch through D
        output = netD(real_images).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=DEVICE)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        del output
        del label
        del noise
        del fake

        return errD.item(), D_x, D_G_z1

def train_generator():
        netG.zero_grad()
        # Generate batch of latent vectors
        noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=DEVICE)
        # Generate fake image batch with G
        fake = netG(noise)
        label = torch.full((BATCH_SIZE,), real_label, dtype=torch.float, device=DEVICE)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        del output
        del fake
        del label
        del noise

        return errG.item(), D_G_z2

print("Starting Training Loop...")
# For each epoch
#load_checkpoint(netG, filename="generator.pkl")
#load_checkpoint(netD, filename="discriminator.pkl")

for epoch in range(EPOCHS):
    torch.cuda.empty_cache()
    # For each batch in the dataloader
    for data in tqdm(dl):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        loss_d, real_score, fake_score1 = train_discriminator(data)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        loss_g, fake_score2 = train_generator()

            # Log losses & scores (last batch)
    print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f} / {:.4}".format(epoch+1, EPOCHS, loss_g, loss_d, real_score, fake_score1, fake_score2))
    save_samples(epoch+1, fixed_noise)
    save_checkpoint(netG, filename="generator.pkl")
    save_checkpoint(netD, filename="discriminator.pkl")
    send_image(fake_fname = 'generated-images-{0:0=4d}.png'.format(index), msg = "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f} / {:.4}".format(epoch+1, EPOCHS, loss_g, loss_d, real_score, fake_score1, fake_score2))
