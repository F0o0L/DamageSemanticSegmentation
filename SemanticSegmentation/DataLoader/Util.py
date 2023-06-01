import os
import cv2
import torch
import random
import shutil
# import imgviz
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from torchvision.utils import make_grid
from PIL import Image, ImageFilter, ImageOps


class LabelProcessor:
    """
    LabelProcessor is used to transform labels from RGB images to tensors which are encoded by natural number,
    or decode predict images to RGB images.
    """

    def __init__(self, data=Union[str, np.ndarray, list]) -> None:
        self.colorMap = self.readColorMap(data)
        self.cm2lbl = self.encodeLabelPix(self.colorMap)

    @staticmethod
    def readColorMap(data=Union[str, np.ndarray, list]) -> np.ndarray:
        colorMap = None
        if isinstance(data, str):
            pdLabelColor = pd.read_csv(data, sep=',')
            colorMap = []
            for i in range(len(pdLabelColor.index)):
                temp = pdLabelColor.iloc[i]
                color = [temp['r'], temp['g'], temp['b']]
                colorMap.append(color)
        else:
            colorMap = data
        colorMap = np.array(colorMap)
        return colorMap

    @staticmethod
    def encodeLabelPix(colorMap: np.ndarray) -> np.ndarray:
        cm2lbl = -1 * np.ones(256 ** 3)
        for i in range(colorMap.shape[0]):
            cm2lbl[(colorMap[i, 0] * 256 + colorMap[i, 1]) * 256 + colorMap[i, 2]] = i
        return cm2lbl

    def encodeLabelImg(self, labelImg: np.ndarray) -> np.ndarray:
        labelImg = np.array(labelImg, dtype='int32')
        index = (labelImg[:, :, 0] * 256 + labelImg[:, :, 1]) * 256 + labelImg[:, :, 2]
        return np.array(self.cm2lbl[index], dtype='int64')

    def decodeLabelImages(self,
                          mode: str,
                          encodeLabels: list,
                          numClasses: int,
                          isSave: bool,
                          filePaths: Optional[list] = None,
                          savePath: Optional[str] = None,
                          classNames: Optional[list] = None,
                          originImages: Optional[list] = None) -> list:

        decodeLabels = []
        if mode == 'decodeType1':
            for i, encodeLabel in enumerate(encodeLabels):
                decodeLabel = -1 * np.ones_like(encodeLabel)
                for k in range(numClasses):
                    idx = np.squeeze(np.where(self.cm2lbl == k))
                    decodeLabel[k == encodeLabel] = idx
                t1 = decodeLabel % 256
                temp = decodeLabel // 256
                t2 = temp % 256
                t3 = temp // 256
                decodeLabel = np.stack((t1, t2, t3)).astype(np.uint8)
                decodeLabels.append(decodeLabel)

                if isSave:
                    if filePaths is not None and savePath is not None:
                        if not os.path.exists(savePath):
                            os.mkdir(savePath)
                        cv2.imwrite(savePath + '/pred' + filePaths[i][filePaths[i].rfind('/') + 1:],
                                    np.transpose(decodeLabel, (1, 2, 0)))
                    else:
                        raise ValueError('filePaths and savePath should not be None!')

        # elif mode == 'decodeType2':
        #     for i, z in enumerate(zip(encodeLabels, originImages)):
        #         encodeLabel, originImg = z
        #         decodeLabel = imgviz.label2rgb(
        #             label=encodeLabel,
        #             image=imgviz.rgb2gray(originImg),
        #             font_size=15,
        #             colormap=self.colorMap,
        #             label_names=classNames,
        #             loc="rb"
        #         ).astype(np.uint8)
        #         decodeLabels.append(decodeLabel)
        #
        #         if isSave:
        #             if not os.path.exists(savePath):
        #                 os.mkdir(savePath)
        #             cv2.imwrite(savePath + '/pred' + filePaths[i][filePaths[i].rfind('/') + 1:], decodeLabel)
        return decodeLabels


def plotGridImages(nrow: int,
                   isSave: bool,
                   preds: list,
                   labels: list,
                   images: Optional[list] = None,
                   savePath: Optional[str] = None,
                   idx: Optional[int] = None):
    preds = torch.Tensor(preds)
    labels = torch.Tensor(labels)
    if images is not None:
        images = np.array(images)
        images = images[:, :, :, [2, 1, 0]]
        images = torch.Tensor(images).permute((0, 3, 1, 2))
        result = make_grid(torch.cat([images, preds, labels], dim=0), nrow, pad_value=255)
    else:
        result = make_grid(torch.cat([preds, labels], dim=0), nrow, pad_value=255)
    if isSave:
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        save = result.numpy().transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(savePath + '/composedPred' + str(idx) + '.png', save)
    return result


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': label}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)
        return {'image': img, 'label': label}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img, 'label': label}


class RandomScaleCrop(object):
    def __init__(self, baseSize, cropSize, fill=0):
        self.baseSize = baseSize
        self.cropSize = cropSize
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        # random scale (short edge)
        shortSize = random.randint(int(self.baseSize * 0.5), int(self.baseSize * 2.0))
        w, h = img.size
        if h > w:
            ow = shortSize
            oh = int(1.0 * h * ow / w)
        else:
            oh = shortSize
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)
        # pad crop
        if shortSize < self.cropSize:
            padh = self.cropSize - oh if oh < self.cropSize else 0
            padw = self.cropSize - ow if ow < self.cropSize else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            label = ImageOps.expand(label, border=(0, 0, padw, padh), fill=self.fill)
        # random crop cropSize
        w, h = img.size
        x1 = random.randint(0, w - self.cropSize)
        y1 = random.randint(0, h - self.cropSize)
        img = img.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))
        label = label.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))

        return {'image': img, 'label': label}


class FixedScaleCrop(object):
    def __init__(self, cropSize):
        self.cropSize = cropSize

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        w, h = img.size
        if w > h:
            oh = self.cropSize
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.cropSize
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.cropSize) / 2.))
        y1 = int(round((h - self.cropSize) / 2.))
        img = img.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))
        label = label.crop((x1, y1, x1 + self.cropSize, y1 + self.cropSize))

        return {'image': img, 'label': label}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        assert img.size == label.size

        img = img.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        return {'image': img, 'label': label}


def readFile(path):
    fileList = os.listdir(path)
    filePathList = [Path(os.path.join(path, image)).as_posix() for image in fileList]
    filePathList.sort()
    return filePathList


def modifyLabels(path: str,
                 savePath: str) -> None:
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    labels = readFile(path=path)
    unchangedColors = [[0, 0, 0], [255, 0, 0]]
    for lab in labels:
        label = Image.open(lab).convert('RGB')
        label = np.array(label)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                flag = True
                for unchangedColor in unchangedColors:
                    if (label[i, j] == unchangedColor).all():
                        flag = False
                if flag:
                    label[i, j] = [0, 0, 0]
                if (label[i, j] == [255, 0, 0]).all():
                    label[i, j] = [0, 128, 0]
        cv2.imwrite(savePath + '/' + lab[lab.rfind('/') + 1:], label[:, :, [2, 1, 0]])


def modifyLabels1(path: str,
                  savePath: str) -> None:
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    labels = readFile(path=path)
    for lab in labels:
        label = Image.open(lab).convert('RGB')
        label = np.array(label)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if not (label[i, j] == [0, 0, 0]).all():
                    if label[i, j, 0] < 32 and label[i, j, 1] < 32:
                        label[i, j] = [0, 0, 0]
                    else:
                        if label[i, j, 0] >= label[i, j, 1]:
                            label[i, j] = [128, 0, 0]
                        else:
                            label[i, j] = [0, 128, 0]
        cv2.imwrite(savePath + '/' + lab[lab.rfind('/') + 1:], label[:, :, [2, 1, 0]])


def changeFileName(path: list,
                   savePath: list,
                   startNum: int) -> None:
    for i in range(len(savePath)):
        if not os.path.exists(savePath[i]):
            os.mkdir(savePath[i])
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    for i, (image, label) in enumerate(zip(images, labels)):
        shutil.copy(image, savePath[0] + '/' + str(startNum + i).zfill(4) + '.png')
        shutil.copy(label, savePath[1] + '/' + str(startNum + i).zfill(4) + '_L.png')


def changeFileName1(path, savePath, startNum):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    images = readFile(path=path)
    for i, image in enumerate(images):
        shutil.copy(image, savePath + '/' + str(startNum + i).zfill(4) + '.png')


def changeFileName2(path: list,
                    savePath: list):
    for i in range(len(savePath)):
        if not os.path.exists(savePath[i]):
            os.mkdir(savePath[i])
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    for image, label in zip(images, labels):
        shutil.copy(image, savePath[0] + '/' + image[image.rfind('/') + 1:].replace('(', '_').replace(')', ''))
        shutil.copy(label, savePath[1] + '/' + label[label.rfind('/') + 1:].replace('(', '_').replace(')', ''))


def cropImages(path: list,
               savePath: list,
               cropSize: int) -> None:
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    for i in range(len(savePath)):
        if not os.path.exists(savePath[i]):
            os.mkdir(savePath[i])
    for img, lab in zip(images, labels):
        image = Image.open(img)
        label = Image.open(lab).convert('RGB')
        w, h = image.size
        if min(w, h) < cropSize:
            if w < h:
                h = int(1.0 * h * cropSize / w)
                w = cropSize
            else:
                w = int(1.0 * w * cropSize / h)
                h = cropSize
            image = image.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)
        i = 0
        j = 0
        index = 1
        while i < w:
            while j < h:
                tempi = i
                tempj = j
                if w - tempi < cropSize:
                    tempi = w - cropSize - 1
                if h - tempj < cropSize:
                    tempj = h - cropSize - 1
                subImage = image.crop((tempi, tempj, tempi + cropSize, tempj + cropSize))
                subLabel = label.crop((tempi, tempj, tempi + cropSize, tempj + cropSize))
                cv2.imwrite(
                    savePath[0] + '/' + img[img.rfind('/') + 1:][:img[img.rfind('/') + 1:].find('.')] + '_' + str(
                        index).zfill(4) + '.png', np.array(subImage)[:, :, [2, 1, 0]])
                cv2.imwrite(
                    savePath[1] + '/' + lab[lab.rfind('/') + 1:][:lab[lab.rfind('/') + 1:].find('.')] + '_' + str(
                        index).zfill(4) + '.png', np.array(subLabel)[:, :, [2, 1, 0]])
                index += 1
                j += cropSize
            i += cropSize
            j = 0


def deleteSubImages(path: list,
                    savePath: list) -> None:
    for i in range(len(savePath)):
        if not os.path.exists(savePath[i]):
            os.mkdir(savePath[i])
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    for img, lab in zip(images, labels):
        image = Image.open(img)
        label = Image.open(lab).convert('RGB')
        image = np.array(image)
        label = np.array(label)
        if not (label == 0).all():
            cv2.imwrite(savePath[0] + '/' + img[img.rfind('/') + 1:], np.array(image)[:, :, [2, 1, 0]])
            cv2.imwrite(savePath[1] + '/' + lab[lab.rfind('/') + 1:], np.array(label)[:, :, [2, 1, 0]])


def divideImages(proportion: list,
                 imagesNum: int,
                 path: list,
                 savePath1: list,
                 savePath2: list) -> None:
    for i in range(len(savePath1)):
        if not os.path.exists(savePath1[i]):
            os.mkdir(savePath1[i])
    for i in range(len(savePath2)):
        if not os.path.exists(savePath2[i]):
            os.mkdir(savePath2[i])
    np.random.seed(1)
    for s in savePath1:
        if not os.path.exists(s):
            os.mkdir(s)
    for s in savePath2:
        if not os.path.exists(s):
            os.mkdir(s)
    proportion = np.array(proportion)
    proportion = proportion / np.sum(proportion)
    proportion = np.floor(imagesNum * proportion).astype(np.int64)
    proportion[0] += imagesNum - np.sum(proportion)
    images = np.arange(1, imagesNum + 1)
    afterDivide = []
    for i in proportion:
        temp = np.random.choice(images, i, replace=False)
        afterDivide.append(temp)
        for j in temp:
            images = np.delete(images, np.where(images == j))
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    for index, prop in enumerate(afterDivide):
        for i in prop:
            for image, label in zip(images, labels):
                img = int(image[image.rfind('/') + 1:][0:4])
                if img == i:
                    shutil.copy(image, savePath1[index])
                    shutil.copy(label, savePath2[index])


def readVideo(path, timeF, savePath, startNum):
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    videoCapture = cv2.VideoCapture(path)
    success, frame = videoCapture.read()
    i = 0
    j = 0
    while success:
        i = i + 1
        if i % timeF == 0:
            j = j + 1
            cv2.imwrite(savePath + '/' + str(startNum + j) + '.png', frame)
            print('save image:', i)
        success, frame = videoCapture.read()


def extractImages(path: list, savePath: list, extractList: list):
    for i in range(len(savePath)):
        if not os.path.exists(savePath[i]):
            os.mkdir(savePath[i])
    images = readFile(path=path[0])
    labels = readFile(path=path[1])
    temp1 = np.array(images)[np.array(extractList) - 1]
    temp2 = np.array(labels)[np.array(extractList) - 1]
    for image, label in zip(temp1, temp2):
        shutil.move(image, savePath[0])
        shutil.move(label, savePath[1])


if __name__ == '__main__':
    # modifyLabels(path='D:/BaiduYunDownload/APESS2018/1_image-wise_labels',
    #              savePath='D:/BaiduYunDownload/data/temp1')
    # modifyLabels1(path='D:/BaiduYunDownload/temp9',
    #               savePath='D:/BaiduYunDownload/temp1')
    # changeFileName(['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/temp/BeforeDivide/images',
    #                 'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/temp/BeforeDivide/labels'],
    #                ['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/temp/BeforeDivide/images1',
    #                 'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/temp/BeforeDivide/labels1'],
    #                startNum=1)
    # changeFileName1('D:/BaiduYunDownload/temp5','D:/BaiduYunDownload/temp6',237)
    # cropImages(['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/train',
    #             'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/train_labels'],
    #            ['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/train',
    #             'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/train_labels'],
    #            512)
    # deleteSubImages(['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/train',
    #                  'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/train_labels'],
    #                 ['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/train',
    #                  'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/train_labels'])
    # divideImages(proportion=[0.8, 0.1, 0.1],
    #              imagesNum=236,
    #              path=['D:/BaiduYunDownload/data/images', 'D:/BaiduYunDownload/data/labels'],
    #              savePath1=['D:/BaiduYunDownload/data/images4/train',
    #                         'D:/BaiduYunDownload/data/images4/val',
    #                         'D:/BaiduYunDownload/data/images4/test'],
    #              savePath2=['D:/BaiduYunDownload/data/labels4/train_labels',
    #                         'D:/BaiduYunDownload/data/labels4/val_labels',
    #                         'D:/BaiduYunDownload/data/labels4/test_labels'])
    # readVideo('D:/BaiduYunDownload/Cylinder.mp4',15,'D:/BaiduYunDownload/temp4',0)
    # extractImages(path=['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/train',
    #                     'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/train_labels'],
    #               savePath=['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/val',
    #                         'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/val_labels'],
    #               extractList=[20, 27, 29, 32, 38, 40, 45, 47, 67, 73, 90, 92, 95, 100, 102, 103, 107, 115, 119, 122,
    #                            146, 175, 180, 192, 195, 197, 205, 210, 213, 218, 222, 226, 229, 232, 241, 248, 249, 250,
    #                            260, 267])
    # changeFileName2(['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/val',
    #                  'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/val_labels'],
    #                 ['F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/val1',
    #                  'F:/Develop/PythonWorkspace/temp/CrackAndCorrosion/AfterDeleteSubImages/val_labels1'])
    lp1 = [[128, 0, 0],
           [0, 128, 0],
           [0, 0, 0]]
    t = [0, 0, 0]
    lp = LabelProcessor(data=lp1)
    labels = readFile('F://PythonWorkspace//temp//CrackAndCorrosion//BeforeDivide//labels')
    for i in range(len(labels)):
        label = Image.open(labels[i]).convert('RGB')
        label = lp.encodeLabelImg(label)
        t[0] += np.count_nonzero(label == 0)
        t[1] += np.count_nonzero(label == 1)
        t[2] += np.count_nonzero(label == 2)
    print(t)
