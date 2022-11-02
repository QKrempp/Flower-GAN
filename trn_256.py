import sys
import os
import gc

import torch
import torch.nn as nn

from torchvision.utils import save_image

from img_256 import DataLoader
from net_256 import Generator, Discriminator
from Telegram_BOT import send_image

data_dir = 'data/256/'
device = 'cuda'
sample_dir = 'generated/'

batch_size = 100
epochs = 30000
lr_gen = 2e-3
lr_dis = 1e-3

nz = 64
nc = 3
ngf = 50
ndf = 50

real_label = 1
fake_label = 0

train_dis_nb = 1
train_gen_nb = 2

lr_decay_nb = 10

def save_sample(epoch, gen, fixed_noise):
    os.makedirs(sample_dir, exist_ok=True)
    fake_img = gen(fixed_noise)
    fake_nam = "generated_{0:0=4d}.png".format(epoch)
    save_image(fake_img, os.path.join(sample_dir, fake_nam), nrow=4)
    del fake_img
    return fake_nam

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

def build_grad_dis(dis, dis_optim, gen, criterion, real_batch):

    # Preparing data
    #dis.zero_grad()
    label = torch.ones((real_batch.shape[0], 1, 1, 1), device=device)

    # Training on real images
    output = dis(real_batch)
    r_output_real = output.mean().item()
    loss_dis_real = criterion(output, real_label * label)
    loss_dis_real.backward()

    # Training on fake images
    noise = torch.randn((real_batch.shape[0], nz, 1, 1), device=device)
    fake_batch = gen(noise)
    output = dis(fake_batch.detach())
    r_output_fake = 1 - output.mean().item()
    loss_dis_fake = criterion(output, fake_label * label)
    loss_dis_fake.backward()

    nn.utils.clip_grad_norm_(dis.parameters(), 5)
    r_loss_dis = loss_dis_real.mean().item() + loss_dis_fake.mean().item()

    #dis_optim.step()

    # Freeing memory
    del label
    del real_batch
    del fake_batch
    del output
    del loss_dis_real
    del loss_dis_fake
    gc.collect()

    return r_loss_dis, r_output_real, r_output_fake
    
def build_grad_gen(gen, gen_optim, dis, criterion):

    # Preparing data
    #gen.zero_grad()
    noise = torch.randn((batch_size, nz, 1, 1), device=device)
    label = torch.ones((batch_size, 1, 1, 1), device=device)

    # Training generator
    fake_batch = gen(noise)
    output = dis(fake_batch)
    r_output_gen = 1 - output.mean().item()
    loss_gen = criterion(output, real_label * label)
    loss_gen.backward()

    nn.utils.clip_grad_norm_(gen.parameters(), 5)
    r_loss_gen = loss_gen.mean().item()

    #gen_optim.step()

    # Freeing memory
    del label
    del fake_batch
    del output
    del loss_gen
    gc.collect()

    return r_loss_gen, r_output_gen

def train(dis, dis_optim, gen, gen_optim, criterion, data_loader):

    r_loss_dis, r_output_real, r_output_fake, r_loss_gen, r_output_gen = 0, 0, 0, 0, 0

    dis.zero_grad()
    for _ in range(train_dis_nb):
        r_loss_dis_tmp, r_output_real_tmp, r_output_fake_tmp = build_grad_dis(dis, dis_optim, gen, criterion, data_loader.get_batch().to(device))
        r_loss_dis += r_loss_dis_tmp / train_dis_nb
        r_output_real += r_output_real_tmp / train_dis_nb
        r_output_fake += r_output_fake_tmp / train_dis_nb
    dis_optim.step()
    
    gen.zero_grad()
    for _ in range(train_gen_nb):
        r_loss_gen_tmp, r_output_gen_tmp = build_grad_gen(gen, gen_optim, dis, criterion)
        r_loss_gen += r_loss_gen_tmp / train_gen_nb
        r_output_gen += r_output_gen_tmp / train_gen_nb
    gen_optim.step()

    return r_loss_dis, r_loss_gen, r_output_real, r_output_fake, r_output_gen

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def training():
    dl = DataLoader(data_dir, batch_size)

    gen = Generator(nz, nc, ngf).to(device)
    gen.apply(weights_init)
    dis = Discriminator(nc, ndf).to(device)
    dis.apply(weights_init)

    gen_optim = torch.optim.Adam(gen.parameters(), lr=lr_gen / train_gen_nb, betas=(0.5, 0.999))
    gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optim, gamma = 0.5, verbose = True)
    dis_optim = torch.optim.Adam(dis.parameters(), lr=lr_dis / train_dis_nb, betas=(0.5, 0.999))
    dis_scheduler = torch.optim.lr_scheduler.ExponentialLR(dis_optim, gamma = 0.5, verbose = True)
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    
    for e in range(epochs):
        loss_dis, loss_gen, out_real, out_fake, out_gen = train(dis, dis_optim, gen, gen_optim, criterion, dl)
        sys.stdout.write("\rEpoch: [%d/%d]\t Discriminator loss: %.4f\t Generator loss: %.4f, Score réel (dis): %.2f / 1, Score fake (dis): %.2f / 1, Score fake (gen): %.2f / 1"%(e + 1, epochs, loss_dis, loss_gen, out_real, out_fake, out_gen))
        sys.stdout.flush()

        if (e + 1) % 100 == 0:
            send_image(fake_fname=save_sample(e, gen, fixed_noise), msg="Epoch [{}/{}], loss_d: {:.4f}, loss_g: {:.4f}".format(e + 1, epochs, loss_dis, loss_gen))
            save_checkpoint(gen, filename="gen.pkl")
            save_checkpoint(dis, filename="dis.pkl")

        if e > 0 and e % (epochs // lr_decay_nb) == 0:
            gen_scheduler.step()
            dis_scheduler.step()

if __name__ == "__main__":
    training()
