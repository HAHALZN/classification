import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(image_list, delimiter, args):
    if len(delimiter) != 1 or len(image_list[-1].split(delimiter)) != 2:
        raise (RuntimeError("the provided delimiter is incorrect: {}".format(delimiter)))
    if args.dx == 'RDR':
        images = [(val.split(delimiter)[0], 0 if int(val.split(delimiter)[1]) < 2 else 1) for val in image_list]
    elif args.dx == 'PDR':
        images = [(val.split(delimiter)[0], 0 if int(val.split(delimiter)[1]) < 4 else 1) for val in image_list]
    elif args.dx == 'NORM':
        images = [(val.split(delimiter)[0], 0 if int(val.split(delimiter)[1]) == 0 else 1) for val in image_list]
    else:
        images = [(val.split(delimiter)[0], int(val.split(delimiter)[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def l_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


class ImageList(Dataset):
    def __init__(self, image_list, args, delimiter=",", transform=None, target_transform=None, mode="RGB"):
        if len(image_list) == 0:
            raise (RuntimeError("no image found, please check the image_list. "))
        imgs = make_dataset(image_list, delimiter, args)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = self._get_num_classes()
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

    def _get_num_classes(self):
        return pd.DataFrame(self.get_labels()).nunique()[0]

    def get_labels(self):
        return [self.imgs[i][1] for i in range(len(self.imgs))]
