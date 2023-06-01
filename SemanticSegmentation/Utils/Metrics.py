import numpy as np
from typing import Dict


class Evaluator:
    def __init__(self, numClasses):
        self.numClasses = numClasses
        self.confusion = np.zeros((numClasses, numClasses), dtype=np.int64)

    def calcSemanticSegmentationConfusion(self,
                                          predLabels: list,
                                          gtLabels: list) -> None:
        predLabels = iter(predLabels)
        gtLabels = iter(gtLabels)
        for predLabel, gtLabel in zip(predLabels, gtLabels):
            if predLabel.ndim != 2 or gtLabel.ndim != 2:
                raise ValueError('label contains 2 dimensions')
            predLabel = predLabel.flatten()
            gtLabel = gtLabel.flatten()

            lbMax = np.max((predLabel, gtLabel))
            if lbMax >= self.numClasses:
                expandedConfusion = np.zeros((lbMax + 1, lbMax + 1), dtype=np.int64)
                expandedConfusion[0:self.numClasses, 0:self.numClasses] = self.confusion

                self.numClasses = lbMax + 1
                self.confusion = expandedConfusion
            mask = gtLabel >= 0
            tempLabel = self.numClasses * gtLabel[mask].astype(int) + predLabel[mask]
            tempLabel = np.bincount(tempLabel, minlength=self.numClasses ** 2)
            self.confusion += tempLabel.reshape((self.numClasses, self.numClasses))
        for iter_ in (predLabels, gtLabels):
            if next(iter_, None) is not None:
                raise ValueError('Length of input iterables need to be same')

    def traverseConfusion(self):
        for i in range(self.confusion.shape[0]):
            s = ''
            for j in range(self.confusion.shape[1]):
                s = s + str(self.confusion[i, j]) + ','
            print(s)

    def calcSemanticSegmentationIoU(self) -> Dict:
        iouDenominator = (self.confusion.sum(axis=1) + self.confusion.sum(axis=0)) - np.diag(self.confusion)
        iou = np.diag(self.confusion) / iouDenominator
        # last number of iou maybe txt
        # return iou[:-1]
        miou = np.nanmean(iou)
        freq = np.sum(self.confusion, axis=1) / np.sum(self.confusion)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()

        return {'miou': miou,
                'fwiou': fwiou,
                'iou': iou}

    def calcSemanticSegmentationAcc(self) -> Dict:
        pixelAcc = np.sum(np.diag(self.confusion) / self.confusion.sum())
        classAcc = np.diag(self.confusion) / (np.sum(self.confusion, axis=1))
        meanClassAcc = np.nanmean(classAcc)
        return {'pixelAcc': pixelAcc,
                'classAcc': classAcc,
                'meanClassAcc': meanClassAcc}


if __name__ == '__main__':
    pLabels = np.random.randint(0, 3, (2, 10, 10))
    gLabels = np.random.randint(0, 3, (2, 10, 10))

    pLabels = [i for i in pLabels]
    gLabels = [i for i in gLabels]

    evaluator = Evaluator(numClasses=3)
    evaluator.calcSemanticSegmentationConfusion(predLabels=pLabels, gtLabels=gLabels)
    evaluator.confusion = np.array([[128333, 0, 752], [0, 0, 0], [374, 0, 132685]])
    iou = evaluator.calcSemanticSegmentationIoU()
    acc = evaluator.calcSemanticSegmentationAcc()
    print(iou)
    print(acc)
