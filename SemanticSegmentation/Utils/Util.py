import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from skimage import morphology
from DataLoader.Util import readFile


# from matplotlib.ticker import MultipleLocator


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculateWeightsLabels(dataset: str,
                           dataLoader: Dataset,
                           numClasses: int) -> np.ndarray:
    # Create an instance from the data loader
    z = np.zeros((numClasses,))
    # Initialize tqdm
    tqdmBatch = tqdm(dataLoader)
    for sample in tqdmBatch:
        tqdmBatch.set_description('Calculating classes weights')
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < numClasses)
        labels = y[mask].astype(np.uint8)
        count = np.bincount(labels, minlength=numClasses)
        z += count
    tqdmBatch.close()
    totalFrequency = np.sum(z)
    classWeights = []
    for frequency in z:
        classWeight = 1 / (np.log(1.02 + (frequency / totalFrequency)))
        classWeights.append(classWeight)
    ret = np.array(classWeights)

    from pathlib import Path
    classesWeightsPath = Path(os.path.join('../DataLoader/Dataset', dataset, dataset + 'ClassesWeights.npy')).as_posix()
    np.save(classesWeightsPath, ret)

    return ret


def plotIteration(path, classNames, withoutBackground=False, confusionPath=None, isPlot=True):
    trainLoss = []
    trainPixelAcc = []
    trainMeanClassAcc = []
    trainMIoU = []
    trainFWIoU = []
    evalLoss = []
    evalPixelAcc = []
    evalMeanClassAcc = []
    evalMIoU = []
    evalFWIoU = []
    classAcc = []
    IoU = []

    with open(path) as f:
        data = f.readlines()
    for line in data:
        t = line.split(',')
        times = t.count('nan')
        if times != 0:
            index = t.index('nan')
            t[index] = '0'
        iou = []
        acc = []
        trainLoss.append(eval(t[0]))
        trainPixelAcc.append(eval(t[1]))
        trainMeanClassAcc.append(eval(t[2]))
        for i in range(len(classNames)):
            acc.append(eval(t[3 + i]))
        trainMIoU.append(eval(t[3 + len(classNames)]))
        trainFWIoU.append(eval(t[4 + len(classNames)]))
        for i in range(len(classNames)):
            iou.append(eval(t[5 + len(classNames) + i]))
        if len(t) > 5 + 2 * len(classNames):
            evalLoss.append(eval(t[5 + 2 * len(classNames)]))
            evalPixelAcc.append(eval(t[6 + 2 * len(classNames)]))
            evalMeanClassAcc.append(eval(t[7 + 2 * len(classNames)]))
            for i in range(len(classNames)):
                acc.append(eval(t[8 + 2 * len(classNames) + i]))
            evalMIoU.append(eval(t[8 + 3 * len(classNames)]))
            evalFWIoU.append(eval(t[9 + 3 * len(classNames)]))
            for i in range(len(classNames)):
                iou.append(eval(t[10 + 3 * len(classNames) + i]))
        classAcc.append(acc)
        IoU.append(iou)
    IoU = np.array(IoU)
    classAcc = np.array(classAcc)

    path = Path(path).as_posix()
    if withoutBackground:
        if confusionPath is None:
            raise ValueError('confusionPath should not be None!')
        else:
            confusion = np.load(confusionPath)
            trainConfusion = confusion['arr_0']
            evalConfusion = confusion['arr_1']
            trainFreq = np.expand_dims(np.sum(trainConfusion, axis=1) / np.sum(trainConfusion), axis=1)[
                        :len(classNames) - 1, :].squeeze()
            evalFreq = np.expand_dims(np.sum(evalConfusion, axis=1) / np.sum(evalConfusion), axis=1)[
                       :len(classNames) - 1, :].squeeze()
            trainFreq = trainFreq / np.sum(trainFreq)
            evalFreq = evalFreq / np.sum(evalFreq)
        trainMIoU = np.mean(IoU[:, :int(IoU.shape[1] / 2 - 1)], axis=1)
        trainMeanClassAcc = np.mean(classAcc[:, :int(classAcc.shape[1] / 2 - 1)], axis=1)
        trainFWIoU = np.sum(trainFreq * IoU[:, :int(IoU.shape[1] / 2 - 1)], axis=1)
        evalMIoU = np.mean(IoU[:, int(IoU.shape[1] / 2):IoU.shape[1] - 1], axis=1)
        evalMeanClassAcc = np.mean(IoU[:, int(classAcc.shape[1] / 2):classAcc.shape[1] - 1], axis=1)
        evalFWIoU = np.sum(evalFreq * IoU[:, int(IoU.shape[1] / 2):IoU.shape[1] - 1], axis=1)
        if isPlot:
            tempPath = path[:path.rfind('/') + 1] + 'withoutBackgroundIter.txt'
            if os.path.exists(tempPath):
                os.remove(tempPath)
            with open(tempPath, "a") as f:
                for i in range(trainMIoU.shape[0]):
                    s = str(trainMeanClassAcc[i]) + ',' + str(trainMIoU[i]) + ',' + str(trainFWIoU[i]) + ',' + str(
                        evalMeanClassAcc[i]) + ',' + str(evalMIoU[i]) + ',' + str(evalFWIoU[i])
                    f.write(s + '\n')
                print('Save at ' + tempPath)

    if isPlot:
        titleFontSize = 24
        legendFontSize = 18
        xyticksFontSize = 18
        plt.figure(dpi=100, figsize=[24, 15.1425])
        ax1 = plt.axes([0.0625, 0.68, 0.25, 0.292])
        ax1.set_title('Loss', fontsize=titleFontSize, y=-0.25)
        ax1.plot(trainLoss, label='Train Loss')
        if len(evalLoss) != 0:
            ax1.plot(evalLoss, label='Eval Loss')
        ax1.legend(fontsize=legendFontSize)
        # ax1.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=xyticksFontSize)
        plt.yticks(fontsize=xyticksFontSize)
        plt.xlabel('epoch', fontsize=xyticksFontSize)
        plt.ylabel('loss', fontsize=xyticksFontSize)
        plt.grid()

        ax2 = plt.axes([0.375, 0.68, 0.25, 0.292])
        ax2.set_title('Pixel Accuracy', fontsize=titleFontSize, y=-0.25)
        ax2.plot(trainPixelAcc, label='Train Accuracy')
        if len(evalPixelAcc) != 0:
            ax2.plot(evalPixelAcc, label='Eval Accuracy')
        ax2.legend(fontsize=legendFontSize)
        # ax2.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=xyticksFontSize)
        plt.yticks(fontsize=xyticksFontSize)
        plt.xlabel('epoch', fontsize=xyticksFontSize)
        plt.ylabel('accuracy', fontsize=xyticksFontSize)
        plt.grid()

        ax3 = plt.axes([0.6875, 0.68, 0.25, 0.292])
        ax3.set_title('FWIoU', fontsize=titleFontSize, y=-0.25)
        ax3.plot(trainFWIoU, label='Train FWIoU')
        if len(evalFWIoU) != 0:
            ax3.plot(evalFWIoU, label='Eval FWIoU')
        ax3.legend(fontsize=legendFontSize)
        # ax3.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=xyticksFontSize)
        plt.yticks(fontsize=xyticksFontSize)
        plt.xlabel('epoch', fontsize=xyticksFontSize)
        plt.ylabel('IoU', fontsize=xyticksFontSize)
        plt.grid()

        ax4 = plt.axes([0.0625, 0.118, 0.40625, 0.474])
        ax4.set_title('Mean Class Accuracy', fontsize=titleFontSize, y=-0.15)
        ax4.plot(trainMeanClassAcc, label='Train Mean Class Accuracy')
        if len(evalMeanClassAcc):
            ax4.plot(evalMeanClassAcc, label='Eval Mean Class Accuracy')
        ax4.legend(fontsize=legendFontSize)
        # ax4.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=xyticksFontSize)
        plt.yticks(fontsize=xyticksFontSize)
        plt.xlabel('epoch', fontsize=xyticksFontSize)
        plt.ylabel('accuracy', fontsize=xyticksFontSize)
        plt.grid()

        ax5 = plt.axes([0.53125, 0.118, 0.40625, 0.474])
        ax5.set_title('MIoU', fontsize=titleFontSize, y=-0.15)
        ax5.plot(trainMIoU, label='Train MIoU')
        if len(evalMIoU):
            ax5.plot(evalMIoU, label='Eval MIoU')
        ax5.legend(fontsize=legendFontSize)
        # ax5.yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(fontsize=xyticksFontSize)
        plt.yticks(fontsize=xyticksFontSize)
        plt.xlabel('epoch', fontsize=xyticksFontSize)
        plt.ylabel('IoU', fontsize=xyticksFontSize)
        plt.grid()

        if withoutBackground:
            path1 = path[:path.rfind('/') + 1] + 'summaryWithoutBackground.png'
        else:
            path1 = path[:path.rfind('/') + 1] + 'summary.png'
        plt.savefig(path1)
        # plt.show()
        print('Save at ' + path1)
        plt.close()

        plt.figure(dpi=100, figsize=[(8 * 4 / 3 + 0.5) * len(classNames), 16])
        for i in range(len(classNames)):
            temp = (1 - (0.1 * 2 + 0.05 * (len(classNames) - 1))) / len(classNames)
            ax = plt.axes([0.1 + i * (temp + 0.05), 0.075, temp, 0.4])
            ax.set_title(classNames[i] + ' IoU', fontsize=titleFontSize, y=-0.15)
            ax.plot(IoU[:, i], label='Train IoU')
            if IoU.shape[1] > len(classNames):
                ax.plot(IoU[:, i + len(classNames)], label='Eval IoU')
            # ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.legend(fontsize=legendFontSize)
            plt.xticks(fontsize=xyticksFontSize)
            plt.yticks(fontsize=xyticksFontSize)
            plt.xlabel('epoch', fontsize=xyticksFontSize)
            plt.ylabel('IoU', fontsize=xyticksFontSize)
            plt.grid()
        for i in range(len(classNames)):
            temp = (1 - (0.1 * 2 + 0.05 * (len(classNames) - 1))) / len(classNames)
            ax = plt.axes([0.1 + i * (temp + 0.05), 0.575, temp, 0.4])
            ax.set_title(classNames[i] + ' Accuracy', fontsize=titleFontSize, y=-0.15)
            ax.plot(classAcc[:, i], label='Train Accuracy')
            if classAcc.shape[1] > len(classNames):
                ax.plot(classAcc[:, i + len(classNames)], label='Eval Accuracy')
            # ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.legend(fontsize=legendFontSize)
            plt.xticks(fontsize=xyticksFontSize)
            plt.yticks(fontsize=xyticksFontSize)
            plt.xlabel('epoch', fontsize=xyticksFontSize)
            plt.ylabel('Accuracy', fontsize=xyticksFontSize)
            plt.grid()
        path2 = path[:path.rfind('/') + 1] + 'AccuracyAndIoU.png'
        plt.savefig(path2)
        print('Save at ' + path2)
        plt.close()

    return trainLoss, trainPixelAcc, trainMeanClassAcc, trainMIoU, trainFWIoU, evalLoss, evalPixelAcc, evalMeanClassAcc, evalMIoU, evalFWIoU, classAcc, IoU


def plotIteration1(iterations, models, classNames, confusionPaths, savePath, styles):
    import matplotlib as mpl
    import pandas as pd
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['font.serif'] = ['Times New Roman']
    # mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

    trainPixelAccs = []
    trainMeanClassAccs = []
    trainMIoUs = []
    trainFWIoUs = []
    evalPixelAccs = []
    evalMeanClassAccs = []
    evalMIoUs = []
    evalFWIoUs = []
    data = pd.DataFrame()
    for i, (iteration, confusionPath) in enumerate(zip(iterations, confusionPaths)):
        itera = plotIteration(iteration, classNames, withoutBackground=True, confusionPath=confusionPath, isPlot=False)
        trainPixelAccs.append(itera[1])
        trainMeanClassAccs.append(itera[2])
        trainMIoUs.append(itera[3])
        trainFWIoUs.append(itera[4])
        evalPixelAccs.append(itera[6])
        evalMeanClassAccs.append(itera[7])
        evalMIoUs.append(itera[8])
        evalFWIoUs.append(itera[9])
        print(models[i] + ':trainPixelAcc:' + str(max(itera[1])))
        print(models[i] + ':trainMeanClassAcc:' + str(max(itera[2])))
        print(models[i] + ':trainMIoU:' + str(max(itera[3])))
        print(models[i] + ':trainFWIoU:' + str(max(itera[4])))
        print(models[i] + ':evalPixelAcc:' + str(max(itera[6])))
        print(models[i] + ':evalMeanClassAcc:' + str(max(itera[7])))
        print(models[i] + ':evalMIoU:' + str(max(itera[8])))
        print(models[i] + ':evalFWIoU:' + str(max(itera[9])))
        data.insert(data.shape[1], models[i] + ':trainPixelAcc', itera[1])
        data.insert(data.shape[1], models[i] + ':trainMeanClassAcc', itera[2])
        data.insert(data.shape[1], models[i] + ':trainMIoU', itera[3])
        data.insert(data.shape[1], models[i] + ':trainFWIoU', itera[4])
        data.insert(data.shape[1], models[i] + ':evalPixelAcc', itera[6])
        data.insert(data.shape[1], models[i] + ':evalMeanClassAcc', itera[7])
        data.insert(data.shape[1], models[i] + ':evalMIoU', itera[8])
        data.insert(data.shape[1], models[i] + ':evalFWIoU', itera[9])
    # data.to_csv('test.csv',sep=',')
    titleFontSize = 30
    legendFontSize = 24
    xyticksFontSize = 24
    lw = 3
    y = -0.18
    plt.figure(dpi=100, figsize=[24, 36])
    ax1 = plt.axes([0.1, 0.78, 0.4, 0.2])
    ax1.set_title('PA(train)', fontsize=titleFontSize, y=y)
    for i, trainPixelAcc in enumerate(trainPixelAccs):
        ax1.plot(trainPixelAcc, styles[i], label=models[i], linewidth=lw, color='k')
    ax1.legend(fontsize=legendFontSize)
    # ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('accuracy', fontsize=xyticksFontSize)
    plt.grid()

    ax2 = plt.axes([0.58, 0.78, 0.4, 0.2])
    ax2.set_title('PA(test)', fontsize=titleFontSize, y=y)
    for i, evalPixelAcc in enumerate(evalPixelAccs):
        ax2.plot(evalPixelAcc, styles[i], label=models[i], linewidth=lw, color='k')
    ax2.legend(fontsize=legendFontSize)
    # ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('accuracy', fontsize=xyticksFontSize)
    plt.grid()

    ax3 = plt.axes([0.1, 0.54, 0.4, 0.2])
    ax3.set_title('MPA(train)', fontsize=titleFontSize, y=y)
    for i, trainMeanClassAcc in enumerate(trainMeanClassAccs):
        ax3.plot(trainMeanClassAcc, styles[i], label=models[i], linewidth=lw, color='k')
    ax3.legend(fontsize=legendFontSize)
    # ax3.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('accuracy', fontsize=xyticksFontSize)
    plt.grid()

    ax4 = plt.axes([0.58, 0.54, 0.4, 0.2])
    ax4.set_title('MPA(test)', fontsize=titleFontSize, y=y)
    for i, evalMeanClassAcc in enumerate(evalMeanClassAccs):
        ax4.plot(evalMeanClassAcc, styles[i], label=models[i], linewidth=lw, color='k')
    ax4.legend(fontsize=legendFontSize)
    # ax4.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('accuracy', fontsize=xyticksFontSize)
    plt.grid()

    ax5 = plt.axes([0.1, 0.30, 0.4, 0.2])
    ax5.set_title('MIoU(train)', fontsize=titleFontSize, y=y)
    for i, trainMIoU in enumerate(trainMIoUs):
        ax5.plot(trainMIoU, styles[i], label=models[i], linewidth=lw, color='k')
    ax5.legend(fontsize=legendFontSize)
    # ax5.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('IoU', fontsize=xyticksFontSize)
    plt.grid()

    ax6 = plt.axes([0.58, 0.30, 0.4, 0.2])
    ax6.set_title('MIoU(test)', fontsize=titleFontSize, y=y)
    for i, evalMIoU in enumerate(evalMIoUs):
        ax6.plot(evalMIoU, styles[i], label=models[i], linewidth=lw, color='k')
    ax6.legend(fontsize=legendFontSize)
    # ax6.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('IoU', fontsize=xyticksFontSize)
    plt.grid()

    ax7 = plt.axes([0.1, 0.06, 0.4, 0.2])
    ax7.set_title('FWIoU(train)', fontsize=titleFontSize, y=y)
    for i, trainFWIoU in enumerate(trainFWIoUs):
        ax7.plot(trainFWIoU, styles[i], label=models[i], linewidth=lw, color='k')
    ax7.legend(fontsize=legendFontSize)
    # ax7.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('IoU', fontsize=xyticksFontSize)
    plt.grid()

    ax8 = plt.axes([0.58, 0.06, 0.4, 0.2])
    ax8.set_title('FWIoU(test)', fontsize=titleFontSize, y=y)
    for i, evalFWIoU in enumerate(evalFWIoUs):
        ax8.plot(evalFWIoU, styles[i], label=models[i], linewidth=lw, color='k')
    ax8.legend(fontsize=legendFontSize)
    # ax8.yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xticks(fontsize=xyticksFontSize)
    plt.yticks(fontsize=xyticksFontSize)
    plt.xlabel('epoch', fontsize=xyticksFontSize)
    plt.ylabel('IoU', fontsize=xyticksFontSize)
    plt.grid()

    plt.savefig(savePath + '/summary1.svg', format='svg')
    print('Save at ' + savePath + '/summary1.svg', )
    plt.close()


def transformPth(oldFilePath, newFilePath):
    torch.save(torch.load(oldFilePath), newFilePath, _use_new_zipfile_serialization=False)
    print('Save at ' + newFilePath)


def extractParameters(checkpoint: str):
    if not os.path.isfile(checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'".format(checkpoint))
    model = torch.load(checkpoint)
    if 'stateDict' in model:
        parameters = model['stateDict']
    else:
        parameters = model
    checkpoint = Path(checkpoint).as_posix()
    path = checkpoint[:checkpoint.rfind('/') + 1] + 'parameters.pth.tar'
    torch.save(parameters, path, _use_new_zipfile_serialization=False)
    print('Save at ' + path)


class MorphologicalFeatureExtraction:
    def __init__(self, paths):
        self.image = cv2.imread(paths[0])
        self.label = cv2.imread(paths[1])

    def __call__(self, savePath='ttt.png'):
        self.mark(1)
        self.mark(2)
        cv2.imwrite(savePath, self.image)

    @staticmethod
    def fastConnectedComponent(imgMat):
        label = 2
        # 用于获取周围元素的掩码
        surroundMask = np.array([[1, 1, 1],
                                 [1, 0, 0],
                                 [0, 0, 0]], np.int8)
        treeArr = np.zeros(imgMat.shape[0] * imgMat.shape[1], dtype=np.int8)
        for row in range(1, imgMat.shape[0] - 1):
            for col in range(1, imgMat.shape[1] - 1):
                if imgMat[row, col] == 0:
                    continue

                surroundElem = surroundMask * imgMat[row - 1:row + 2, col - 1:col + 2]
                surroundLabels = surroundElem[surroundElem != 0]

                if surroundLabels.shape[0] != 0:
                    minSurroundLabel = surroundLabels.min()
                    if minSurroundLabel != 0 and minSurroundLabel != 1:  # 周围存在被标记过的数据
                        imgMat[row, col] = minSurroundLabel
                        treeArr[surroundLabels[surroundLabels != minSurroundLabel]] = minSurroundLabel
                        continue

                imgMat[row, col] = label
                treeArr[label] = label
                label += 1

        for i in range(2, label):
            k = i
            while k != treeArr[k]:
                k = treeArr[k]
            treeArr[i] = k

        for row in range(imgMat.shape[0]):
            for col in range(imgMat.shape[1]):
                imgMat[row, col] = treeArr[imgMat[row, col]]

        imgMat1 = imgMat.copy()
        temp = []
        while len(imgMat1[imgMat1 == 0]) != imgMat1.shape[0] * imgMat1.shape[1]:
            temp.append(imgMat1.max())
            imgMat1[imgMat1 == imgMat1.max()] = 0

        for i in range(len(temp)):
            imgMat[imgMat == temp[len(temp) - 1 - i]] = 2 + i
        return imgMat

    def getRect(self, imgMat: np.ndarray, channel):
        maxNum = imgMat.max()
        high = []
        low = []
        left = []
        right = []
        morph = []
        for i in range(2, maxNum + 1):
            h = None
            lo = None
            le = None
            r = None
            temp2 = []
            for row in range(0, imgMat.shape[0]):
                temp1 = imgMat[row, :][imgMat[row, :] == i]
                if len(temp1) > 0 and len(temp2) == 0:
                    h = row
                elif len(temp1) == 0 and len(temp2) > 0:
                    lo = row
                temp2 = temp1.copy()
            temp2 = []
            for col in range(0, imgMat.shape[1]):
                temp1 = imgMat[:, col][imgMat[:, col] == i]
                if len(temp1) > 0 and len(temp2) == 0:
                    le = col
                elif len(temp1) == 0 and len(temp2) > 0:
                    r = col
                temp2 = temp1.copy()
            if channel == 1:
                temp = imgMat[h:lo, le:r].copy()
                temp[temp == i] = 1
                temp[temp != 1] = 0
                skel = morphology.medial_axis(temp, return_distance=False)
                length = len(temp[temp * skel == 1])
                width = len(temp[temp == 1]) / (length + 1e-50)
                if length / imgMat.shape[0] > 0 or length / imgMat.shape[1] > 0:
                    self.image[:, :, 0][imgMat == i] = 0
                    self.image[:, :, 1][imgMat == i] = 128
                    self.image[:, :, 2][imgMat == i] = 0
                    morph.append([length, width])
                    high.append(h)
                    low.append(lo)
                    left.append(le)
                    right.append(r)
            if channel == 2:
                if (lo - h) / imgMat.shape[0] > 0 or (r - le) / imgMat.shape[1] > 0:
                    self.image[:, :, 0][imgMat == i] = 0
                    self.image[:, :, 1][imgMat == i] = 0
                    self.image[:, :, 2][imgMat == i] = 128
                    area = len(imgMat[imgMat == i])
                    morph.append(area)
                    high.append(h)
                    low.append(lo)
                    left.append(le)
                    right.append(r)
        return high, low, left, right, morph

    def mark(self, channel, outputPic=True):
        layer = self.label[:, :, channel]
        layer[layer > 0] = 1
        kernel = np.ones((20, 20), np.uint8)
        layer = cv2.morphologyEx(layer, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('s', layer)
        # cv2.waitKey()
        imgMat = self.fastConnectedComponent(layer)
        high, low, left, right, morph = self.getRect(imgMat, channel)
        if outputPic:
            for i, (h, lo, le, r) in enumerate(zip(high, low, left, right)):
                cv2.rectangle(self.image, (le - 10, h - 10), (r + 10, lo + 10),
                              (0 if channel != 0 else 128, 0 if channel != 1 else 128, 0 if channel != 2 else 128), 2)
                if channel == 2:
                    if h - 10 < 0:
                        h = lo + 45
                    cv2.rectangle(self.image,
                                  (le - 10, h - 35),
                                  (le + 100, h - 10),
                                  color=(0, 0, 128),
                                  thickness=-1)
                    cv2.putText(self.image, 'corrosion', (le - 10, h - 25),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(self.image, 'area:' + str(morph[i]), (le - 10, h - 13),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                elif channel == 1:
                    cv2.rectangle(self.image,
                                  (le - 10, h - 60),
                                  (le + 100, h - 10),
                                  color=(0, 128, 0),
                                  thickness=-1)
                    cv2.putText(self.image, 'crack', (le - 10, h - 45),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(self.image, 'length:' + str(morph[i][0]), (le - 10, h - 30),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
                    cv2.putText(self.image, 'width:' + str(round(morph[i][1], 3)), (le - 10, h - 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (255, 255, 255), 1)
        return morph


def getMorph(paths):
    imgPaths = readFile(paths[0])
    labelPaths = readFile(paths[1])
    predpaths = readFile(paths[2])

    for i in range(len(imgPaths)):
        s = imgPaths[i]
        s1 = ''
        crack = MorphologicalFeatureExtraction([imgPaths[i], labelPaths[i]]).mark(1, False)
        for j in range(len(crack)):
            s1 += s
            for k in range(len(crack[j])):
                s1 += ','
                s1 += str(crack[j][k])
            print(s1)
            s1 = ''
        corrosion = MorphologicalFeatureExtraction([imgPaths[i], labelPaths[i]]).mark(2)
        for j in range(len(corrosion)):
            s1 += s
            s1 += ',,,'
            s1 += str(corrosion[j])
            print(s1)
            s1 = ''

        crack = MorphologicalFeatureExtraction([imgPaths[i], predpaths[i]]).mark(1, False)
        for j in range(len(crack)):
            s1 += s
            for k in range(len(crack[j])):
                s1 += ','
                s1 += str(crack[j][k])
            print(s1)
            s1 = ''
        corrosion = MorphologicalFeatureExtraction([imgPaths[i], predpaths[i]]).mark(2)
        for j in range(len(corrosion)):
            s1 += s
            s1 += ',,,'
            s1 += str(corrosion[j])
            print(s1)
            s1 = ''


if __name__ == '__main__':
    # from DataLoader.MyDataLoader import MyDataset
    #
    # trainLoader = MyDataset(action='train',
    #                         filePath=['../../temp/CrackAndCorrosion/AfterDivide/train','../../temp/CrackAndCorrosion/AfterDivide/train_labels'],
    #                         lpFilePath=[[128, 0, 0],[0, 128, 0],[0, 0, 0]],
    #                         baseSize=512,
    #                         cropSize=512,
    #                         outputOriImg=False,
    #                         noDataAugmentation=True)
    #
    # calculateWeightsLabels('CrackAndCorrosion', trainLoader, 3)

    # extractParameters('/Run/CamVid/Deeplab-DRN-ASPP/Experiment1/checkpoint.pth.tar')

    # r = np.random.random(size=(22, 50))
    # for i in range(r.shape[1]):
    #     s = ''
    #     for j in range(r.shape[0]):
    #         s = s + str(r[j, i]) + ','
    #     s = s[:len(s) - 1]
    #     with open('F:/Develop/PythonWorkspace/temp/Run/ttt.txt', "a") as f:
    #         f.write(s + '\n')

    plotIteration(
        'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment0/iter1.txt',
        ['Corrosion', 'Crack', 'Background'],
        withoutBackground=True,
        confusionPath='F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment0/confusion1.npz')

    # plotIteration1(
    #     ['F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-MobileNetV3Large-ASPP/Experiment0/iter.txt',
    #      'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-MobileNetV3Large-ASPP/Experiment1/iter.txt',
    #      'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment0/iter.txt',
    #      'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment1/iter.txt',
    #      'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/FCN/Experiment0/iter.txt'],
    #     ['MobileNetV3(8)', 'MobileNetV3(16)', 'DRN(8)', 'DRN(16)', 'FCN'],
    #     ['Corrosion', 'Crack', 'Background'],
    #     [
    #         'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-MobileNetV3Large-ASPP/Experiment0/confusion.npz',
    #         'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-MobileNetV3Large-ASPP/Experiment1/confusion.npz',
    #         'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment0/confusion.npz',
    #         'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-DRN-ASPP/Experiment1/confusion.npz',
    #         'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/FCN/Experiment0/confusion.npz'],
    #     'F:/PythonWorkspace/temp/Run', styles=['-.', '--', '-', ':', '-'])

    # m = MorphologicalFeatureExtraction(['F:/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/ves/0146.png',
    #                                     'F:/PythonWorkspace/temp/CrackAndCorrosion/AfterDivide/vl/0146_L.png'])

    # MorphologicalFeatureExtraction(
    #     ['F:/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/val/0020_0001.png',
    #      'F:/PythonWorkspace/temp/CrackAndCorrosion/AfterCrop/val_labels/0020_L_0001.png'])()

    # MorphologicalFeatureExtraction(
    #     ['C:/Users/ou878/Desktop/val/0192_0012.png',
    #      'C:/Users/ou878/Desktop/val_labels/0192_L_0012.png'])()

    # getMorph(['C:/Users/ou878/Desktop/train',
    #           'C:/Users/ou878/Desktop/train_label',
    #           'C:/Users/ou878/Desktop/trainpred'])
