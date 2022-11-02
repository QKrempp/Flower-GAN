import random
import torch
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as tt

class DataLoader:

    def __init__(self, data_dir, batch_size, file="files.txt"):
        self.img_list   = open(data_dir + file, 'r').read().splitlines()
        self.img_path   = data_dir
        self.batch_size = batch_size
        self.batch_id   = 0
        self.batch_nb   = len(self.img_list) // self.batch_size
        self.transform  = tt.Compose([tt.ConvertImageDtype(torch.float), tt.Resize(256), tt.CenterCrop(256)])
        #self.transform  = tt.Compose([tt.ConvertImageDtype(torch.float), tt.Resize(256), tt.CenterCrop(256), tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def shuffle(self):
        random.shuffle(self.img_list)

    def get_batch(self):

        if self.batch_id == 0:
            self.shuffle()
        
        img_list        = self.img_list[self.batch_id * self.batch_size:(self.batch_id + 1) * self.batch_size]
        torch_img = []
        for img in img_list:

            try:
                torch_img.append(self.transform(read_image(self.img_path + img, ImageReadMode.RGB).unsqueeze(0)))
            except RuntimeError:
                print("\nImage " + img + " non supportée")

        self.batch_id   = (self.batch_id + 1) % self.batch_nb
        return torch.cat(torch_img)


if __name__ == "__main__":
    dl = DataLoader('data/256/', 128)
    print(dl.get_batch().shape)
