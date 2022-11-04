import torch
import pandas as pd

from tqdm import tqdm
from net_256 import Discriminator
from torchvision.io import read_image, ImageReadMode

import torchvision.transforms as tt

device = "cuda"

img_list   = open("data/256/files.txt", 'r').read().splitlines()

dis = Discriminator(3, 50)
dis.load_checkpoint(device, filename="dis.pkl")
dis.to(device)

torch_transform = tt.Compose([tt.ConvertImageDtype(torch.float), tt.Resize(256), tt.CenterCrop(256)])

dc_values = {}

for img in tqdm(img_list):
    try:
        torch_img = torch_transform(read_image("data/256/" + img, ImageReadMode.RGB).unsqueeze(0))
        dc_values[img] = dis(torch_img.to(device)).item()
    except RuntimeError:
        print("Image " + img + " non supportée")


pd_values = pd.DataFrame.from_dict(dc_values, orient='index')
pd_values.columns = ['confidence']

print(pd_values.describe())

pd_values.to_csv('outliers.csv')

#floor = pd_values.quantile(0.001)
#
#print(float(floor))
#
#id_outliers = pd_values[pd_values['confidence'] < 0.8].index.values
#
#fl_outliers = open("outliers.txt", 'w')
#
#for o in id_outliers:
#    fl_outliers.writelines(o + '\n')
