from torch.utils import data
import os
from PIL import Image
from torchvision.transforms import ToTensor
import numpy
import glob
import random
import numpy as np

def augment(lr, hr, p,  hflip=True, rot=True):
    # def _augment(img):
    #     if hflip: img = img[:, ::-1, :]
    #     if vflip: img = img[::-1, :, :]
    #     if rot90: img = img.transpose(1, 0, 2)
    #     return img

    if random.random() > 0.5 and hflip:
        lr = lr[:, ::-1, :]
        hr = hr[:, ::-1, :]
        p = p[:, ::-1, :]
        # print("hflip")

    if rot:
        rot_rand = random.random()
        if rot_rand > 0.75:
            lr = np.rot90(lr, k=1, axes=(0, 1))
            hr = np.rot90(hr, k=1, axes=(0, 1))
            p = np.rot90(p, k=1, axes=(0, 1))
        elif rot_rand > 0.5:
            lr = np.rot90(lr, k=2, axes=(0, 1))
            hr = np.rot90(hr, k=2, axes=(0, 1))
            p = np.rot90(p, k=2, axes=(0, 1))

        elif rot_rand > 0.25:
            lr = np.rot90(lr, k=3, axes=(0, 1))
            hr = np.rot90(hr, k=3, axes=(0, 1))
            p = np.rot90(p, k=3, axes=(0, 1))
        # print("rot")
    return lr, hr, p
class Data(data.Dataset):
    def __init__(self, root, args, train=False):
        # 返回指定路径下的文件和文件夹列表。
        self.args = args
        self.imgs_HR_path = os.path.join(root, 'HR')
        self.imgs_HR = sorted(
            glob.glob(os.path.join(self.imgs_HR_path, '*.png'))
        )

        if self.args.scale == 8:
            self.imgs_LR_path = os.path.join(root, 'LR')
        elif self.args.scale == 16:
            self.imgs_LR_path = os.path.join(root, 'LR_x16')
        elif self.args.scale == 4:
            self.imgs_LR_path = os.path.join(root, 'LR_x4')

        self.imgs_LR = sorted(
            glob.glob(os.path.join(self.imgs_LR_path, '*.png'))
        )

        if self.args.scale == 8:
            self.imgs_parsing_path = os.path.join(root, 'Es_parsing')
        elif self.args.scale == 16:
            self.imgs_parsing_path = os.path.join(root, 'Es_parsing_x16')
        elif self.args.scale == 4:
            self.imgs_parsing_path = os.path.join(root, 'Es_parsing_x4')

        self.imgs_parsing = sorted(
            glob.glob(os.path.join(self.imgs_parsing_path, '*.png'))
        )
        self.train = train

    def __getitem__(self, item):

        img_path_LR = os.path.join(self.imgs_LR_path, self.imgs_LR[item])
        img_path_HR = os.path.join(self.imgs_HR_path, self.imgs_HR[item])
        img_path_parsing = os.path.join(self.imgs_parsing_path, self.imgs_parsing[item])
        LR = Image.open(img_path_LR)
        HR = Image.open(img_path_HR)
        parsing = Image.open(img_path_parsing)
        HR = numpy.array(HR)
        LR = numpy.array(LR)
        parsing = numpy.array(parsing)
        if self.args.augment and self.train:
            LR, HR, parsing = augment(LR, HR, parsing)

        LR = np.ascontiguousarray(LR)
        HR = np.ascontiguousarray(HR)
        parsing = np.ascontiguousarray(parsing)
        HR = ToTensor()(HR)
        LR = ToTensor()(LR)
        res = ToTensor()(parsing)
        filename = os.path.basename(img_path_HR)

        return LR, HR, res, filename


    def __len__(self):
        return len(self.imgs_HR)

