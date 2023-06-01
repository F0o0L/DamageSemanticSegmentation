import numpy as np
import torch

import DataLoader.Util as Util
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Union
from DataLoader.Util import LabelProcessor, readFile


class MyDataset(Dataset):
    """
    load dataset
    """

    def __init__(self,
                 action: str,
                 filePath: list,
                 lpFilePath: Union[str, np.ndarray, list],
                 baseSize: Union[int, tuple],
                 cropSize: Union[int, tuple],
                 outputOriImg: bool = False,
                 noDataAugmentation: bool = False) -> None:
        self.lp = LabelProcessor(data=lpFilePath)  # an object of class LabelProcessor

        self.action = action

        if len(filePath) != 2:
            raise ValueError('need file paths of images and labels!')
        self.imgFilePath = filePath[0]
        self.labelFilePath = filePath[1]

        self.images = readFile(self.imgFilePath)
        self.labels = readFile(self.labelFilePath)

        self.baseSize = baseSize
        self.cropSize = cropSize

        self.outputOriImg = outputOriImg
        self.noDataAugmentation = noDataAugmentation

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = Image.open(image)
        label = Image.open(label).convert('RGB')

        sample = self.imgTransform(image, label)
        sample = {'img': sample[0],
                  'label': sample[1],
                  'oriImg': sample[2]} if len(sample) == 3 else {'img': sample[0],
                                                                 'label': sample[1]}

        return sample

    def imgTransform(self, image, label):
        oriImg = None
        sample = {'image': image, 'label': label}
        if not self.noDataAugmentation:
            composedTransforms = None
            if self.action == 'train':
                composedTransforms = transforms.Compose([
                    Util.RandomHorizontalFlip(p=0.5),
                    Util.RandomScaleCrop(baseSize=self.baseSize, cropSize=self.cropSize, fill=0),
                    Util.RandomRotate(180),
                    Util.RandomGaussianBlur()
                ])
            elif self.action == 'val':
                composedTransforms = transforms.Compose([
                    Util.FixedScaleCrop(cropSize=self.cropSize)
                ])
            elif self.action == 'test':
                composedTransforms = transforms.Compose([
                    Util.FixedResize(size=self.cropSize)
                ])
            sample = composedTransforms(sample)
        image, label = np.array(sample['image']), np.array(sample['label'])

        if self.outputOriImg:
            oriImg = image
        transformImg = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transformImg(image)
        label = self.lp.encodeLabelImg(label)
        label = torch.from_numpy(label)
        return (image, label, oriImg) if oriImg is not None else (image, label)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import cv2
    from torch.utils.data import DataLoader

    # trainRoot = 'Dataset/CamVid/train'
    # trainLabel = 'Dataset/CamVid/train_labels'
    # valRoot = 'Dataset/CamVid/val'
    # valLabel = 'Dataset/CamVid/val_labels'
    # testRoot = 'Dataset/CamVid/test'
    # testLabel = 'Dataset/CamVid/test_labels'

    # lp = 'Dataset/CamVid/class_dict.csv'

    trainRoot = 'Dataset/CrackAndCorrosion/train'
    trainLabel = 'Dataset/CrackAndCorrosion/train_labels'
    valRoot = 'Dataset/CrackAndCorrosion/val'
    valLabel = 'Dataset/CrackAndCorrosion/val_labels'
    testRoot = 'Dataset/CrackAndCorrosion/test'
    testLabel = 'Dataset/CrackAndCorrosion/test_labels'

    lp = [[128, 0, 0],
          [0, 128, 0],
          [0, 0, 0]]

    train = MyDataset(action='train',
                      filePath=[trainRoot, trainLabel],
                      lpFilePath=lp,
                      baseSize=512,
                      cropSize=512,
                      outputOriImg=False,
                      noDataAugmentation=False)
    trainData = DataLoader(dataset=train,
                           batch_size=1,
                           shuffle=False,
                           num_workers=2)
    for i, s in enumerate(trainData):
        # im = s['img'].numpy().squeeze()
        lll = s['label'].numpy().squeeze()

        # cv2.imshow('sd', im)
        # cv2.imshow('sss', lll)
        # cv2.waitKey()
        a = (lll == 0)
        b = (lll == 1)
        c = (lll == 2)
        d = a + b + c
        print(np.all(d))
