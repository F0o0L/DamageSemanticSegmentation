import os
import time
import torch
import numpy as np
from tqdm import tqdm
from argparse import Namespace
from typing import Tuple, Optional, Dict
from torch.utils.data import DataLoader  # , Dataset
from MyModeling.MyDeeplabV3Plus import DeeplabV3Plus
from MyModeling.FCN import FCN
from DataLoader.MyDataLoader import MyDataset
from DataLoader.Util import plotGridImages
from Utils.Saver import Saver
from Utils.Loss import SegmentationLosses
from Utils.Metrics import Evaluator
from Utils.LearningRateScheduler import LRScheduler
from Utils.Util import seed_torch, calculateWeightsLabels


class Trainer(object):
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.saver = Saver(self.args)
        if not self.args.isTest:
            self.saver.saveExperimentConfig()
        # self.sw = SummaryWriter(log_dir=self.saver.experimentDir + '/Summary', flush_secs=1)

        # load model
        self.model, self.bestPred, self.lastEpoch = self.loadModel()
        # self.sw.add_graph(model=self.model, input_to_model=torch.randn(2, 3, self.args.cropSize, self.args.cropSize))

        # load data
        self.trainLoader, self.valLoader, self.testLoader, self.trainData, self.valData, self.testData = self.loadData()

        # define optimizer
        self.optimizer = torch.optim.Adam(params=[{'params': self.model.parameters(), 'initial_lr': self.args.lr}],
                                          lr=self.args.lr)

        # define loss
        self.segLoss = self.defineLoss()

        # whether to use cuda
        if self.args.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

        # define learning rate scheduler
        self.scheduler = LRScheduler(mode=self.args.lrScheduler,
                                     optimizer=self.optimizer,
                                     numEpochs=self.args.epochs,
                                     lastEpoch=self.lastEpoch)(stepSize=50, gamma=0.5)

    def loadModel(self) -> Tuple[torch.nn.Module, float, int]:
        bestPred = 0.0
        lastEpoch = -1

        model = None
        if self.args.model == 'FCN':
            if self.args.resume is not None:
                model = FCN(numClasses=self.args.numClasses,
                            preTrained=False)
                self.args.preTrained = False
            else:
                model = FCN(numClasses=self.args.numClasses,
                            preTrained=self.args.preTrained)
        elif self.args.model == 'DeeplabV3Plus':
            if self.args.resume is not None:
                model = DeeplabV3Plus(backbone=self.args.backbone,
                                      ASPP=self.args.ASPP,
                                      preTrained=False,
                                      modify=self.args.modify,
                                      dilations=self.args.dilations,
                                      numClasses=self.args.numClasses,
                                      outputStride=self.args.outputStride,
                                      lowLevelOutputStride=self.args.lowLevelOutputStride,
                                      dilated=self.args.dilated)
                self.args.preTrained = False
            else:
                model = DeeplabV3Plus(backbone=self.args.backbone,
                                      ASPP=self.args.ASPP,
                                      preTrained=self.args.preTrained,
                                      modify=self.args.modify,
                                      dilations=self.args.dilations,
                                      numClasses=self.args.numClasses,
                                      outputStride=self.args.outputStride,
                                      lowLevelOutputStride=self.args.lowLevelOutputStride,
                                      dilated=self.args.dilated)

        # whether to resume training previous trained model
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)
            if 'stateDict' in checkpoint:
                model.load_state_dict(checkpoint['stateDict'], strict=False)
            else:
                model.load_state_dict(checkpoint)
            if 'epoch' in checkpoint:
                self.args.startEpoch = checkpoint['epoch']
                lastEpoch = checkpoint['epoch']
            if 'bestPred' in checkpoint:
                bestPred = checkpoint['bestPred']
            if not self.args.isTest:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.args.resume, self.args.startEpoch))
        return model, bestPred, lastEpoch

    def loadData(self):
        trainLoader = None
        valLoader = None
        testLoader = None
        trainData = None
        valData = None
        testData = None
        if self.args.dataset == 'CamVid' or self.args.dataset == 'CrackAndCorrosion':
            trainRoot = None
            trainLabel = None
            valRoot = None
            valLabel = None
            testRoot = None
            testLabel = None
            lp = None
            if self.args.dataset == 'CamVid':
                trainRoot = 'DataLoader/Dataset/CamVid/train'
                trainLabel = 'DataLoader/Dataset/CamVid/train_labels'
                valRoot = 'DataLoader/Dataset/CamVid/val'
                valLabel = 'DataLoader/Dataset/CamVid/val_labels'
                testRoot = 'DataLoader/Dataset/CamVid/test'
                testLabel = 'DataLoader/Dataset/CamVid/test_labels'
                lp = 'DataLoader/Dataset/CamVid/class_dict.csv'
            elif self.args.dataset == 'CrackAndCorrosion':
                trainRoot = 'DataLoader/Dataset/CrackAndCorrosion/train'
                trainLabel = 'DataLoader/Dataset/CrackAndCorrosion/train_labels'
                valRoot = 'DataLoader/Dataset/CrackAndCorrosion/val'
                valLabel = 'DataLoader/Dataset/CrackAndCorrosion/val_labels'
                testRoot = 'DataLoader/Dataset/CrackAndCorrosion/test'
                testLabel = 'DataLoader/Dataset/CrackAndCorrosion/test_labels'
                lp = [[128, 0, 0],
                      [0, 128, 0],
                      [0, 0, 0]]
            if self.args.isTest:
                testLoader = MyDataset(
                    action='test',
                    filePath=[testRoot, testLabel],
                    lpFilePath=lp,
                    baseSize=self.args.baseSize,
                    cropSize=self.args.testCropSize,
                    outputOriImg=not self.args.onlyGtAndPred,
                    noDataAugmentation=self.args.noDataAugmentation)
            else:
                trainLoader = MyDataset(
                    action='train',
                    filePath=[trainRoot, trainLabel],
                    lpFilePath=lp,
                    baseSize=self.args.baseSize,
                    cropSize=self.args.cropSize,
                    outputOriImg=False,
                    noDataAugmentation=self.args.noDataAugmentation)
                if not self.args.noVal:
                    valLoader = MyDataset(
                        action='val',
                        filePath=[valRoot, valLabel],
                        lpFilePath=lp,
                        baseSize=self.args.baseSize,
                        cropSize=self.args.testCropSize,
                        outputOriImg=not self.args.onlyGtAndPred,
                        noDataAugmentation=self.args.noDataAugmentation)

        if self.args.isTest:
            testData = DataLoader(dataset=testLoader,
                                  batch_size=self.args.testBatchSize,
                                  num_workers=self.args.workers)
        else:
            if len(trainLoader) % self.args.batchSize == 1:
                trainData = DataLoader(dataset=trainLoader,
                                       batch_size=self.args.batchSize,
                                       shuffle=True,
                                       num_workers=self.args.workers,
                                       drop_last=True)
            else:
                trainData = DataLoader(dataset=trainLoader,
                                       batch_size=self.args.batchSize,
                                       shuffle=True,
                                       num_workers=self.args.workers)
            if not self.args.noVal:
                valData = DataLoader(dataset=valLoader,
                                     batch_size=self.args.testBatchSize,
                                     num_workers=self.args.workers,
                                     shuffle=True)

        return trainLoader, valLoader, testLoader, trainData, valData, testData

    def defineLoss(self) -> SegmentationLosses:
        # whether to use balanced weights
        if self.args.useBalancedWeights:
            classesWeightsPath = os.path.join('DataLoader/Dataset', self.args.dataset,
                                              self.args.dataset + 'ClassesWeights.npy')
            if os.path.isfile(classesWeightsPath):
                weight = np.load(classesWeightsPath)
            else:
                weight = calculateWeightsLabels(dataset=self.args.dataset,
                                                dataLoader=self.trainLoader,
                                                numClasses=self.args.numClasses)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        # an object of SegmentationLosses, we will define loss latter by using this
        segLoss = SegmentationLosses(weight=weight, cuda=self.args.cuda)
        return segLoss

    def training(self, trainData) -> None:
        index = 0
        trainEvaluator = Evaluator(numClasses=self.args.numClasses)
        for epoch in range(self.args.startEpoch, self.args.epochs):
            tbar = tqdm(trainData)
            trainLoss = 0

            net = self.model.train()
            for i, sample in enumerate(tbar):
                trainImg = sample['img'].to(self.device)
                trainLabel = sample['label'].to(self.device)

                out = net(trainImg)
                loss = self.segLoss(out, trainLabel, mode=self.args.lossType, classNum=self.args.numClasses)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                trainLoss += loss.item()
                tbar.set_description('Train Batch:{},Train Loss:{:.3f},Learning Rate:{},Previous Best:{:.3f}'.
                                     format(i, trainLoss / (i + 1), self.optimizer.param_groups[0]['lr'],
                                            self.bestPred))

                predLabel = out.max(dim=1)[1].data.cpu().numpy()
                predLabel = [j for j in predLabel]

                trueLabel = trainLabel.data.cpu().numpy()
                trueLabel = [j for j in trueLabel]

                trainEvaluator.calcSemanticSegmentationConfusion(predLabels=predLabel, gtLabels=trueLabel)
                index += 1
            trainAcc = trainEvaluator.calcSemanticSegmentationAcc()
            trainPixelAcc = trainAcc['pixelAcc']
            trainMeanClassAcc = trainAcc['meanClassAcc']
            trainClassAcc = trainAcc['classAcc']
            trainIoUs = trainEvaluator.calcSemanticSegmentationIoU()
            trainMIoU = trainIoUs['miou']
            trainFWIoU = trainIoUs['fwiou']
            trainIoU = trainIoUs['iou']

            if self.args.lrScheduler == 'reduceLROnPlateau':
                self.scheduler.step(metrics=trainMIoU)
            else:
                self.scheduler.step()

            print()
            print('|Epoch|:{}'.format(epoch))
            print('|Train Loss|:{:.5f}'.format(trainLoss / len(trainData)))
            print('|Train Pixel Acc|:{:.5f}'.format(trainPixelAcc))
            print('|Train Mean Class Acc|:{:.5f}'.format(trainMeanClassAcc))
            print('|Train Mean IoU|:{:.5f}'.format(trainMIoU))
            print('|Train FWIoU|:{:.5f}'.format(trainFWIoU))
            for className, acc, iou in zip(self.args.classNames, trainClassAcc, trainIoU):
                print('|Train ' + className + ' Accuracy|:{:.5f}'.format(acc))
                print('|Train ' + className + ' IoU|:{:.5f}'.format(iou))

            time.sleep(0.1)

            if self.args.noVal:
                # save checkpoint every epoch
                isBest = False
                self.saver.saveCheckpoint({
                    'epoch': epoch + 1,
                    'stateDict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'bestPred': self.bestPred,
                }, isBest)
                self.saver.saveIteration(trainLoss=trainLoss / len(trainData),
                                         trainPixelAcc=trainPixelAcc,
                                         trainMeanClassAcc=trainMeanClassAcc,
                                         trainClassAcc=trainClassAcc,
                                         trainMIoU=trainMIoU,
                                         trainFWIoU=trainFWIoU,
                                         trainIoU=trainIoU,
                                         trainConfusion=trainEvaluator.confusion)
                trainEvaluator.traverseConfusion()
            else:
                if epoch % self.args.evalInterval == (self.args.evalInterval - 1):
                    evalResult = self.validation(
                        valLoader=self.valLoader,
                        valData=self.valData,
                        isSave=True,
                        epoch=epoch,
                        savePath=os.path.join(self.saver.experimentDir, 'pred'))

                    self.saver.saveIteration(trainLoss=trainLoss / len(trainData),
                                             trainPixelAcc=trainPixelAcc,
                                             trainMeanClassAcc=trainMeanClassAcc,
                                             trainClassAcc=trainClassAcc,
                                             trainMIoU=trainMIoU,
                                             trainFWIoU=trainFWIoU,
                                             trainIoU=trainIoU,
                                             trainConfusion=trainEvaluator.confusion,
                                             evalLoss=evalResult['evalLoss'],
                                             evalPixelAcc=evalResult['evalPixelAcc'],
                                             evalMeanClassAcc=evalResult['evalMeanClassAcc'],
                                             evalClassAcc=evalResult['evalClassAcc'],
                                             evalMIoU=evalResult['evalMIoU'],
                                             evalFWIoU=evalResult['evalFWIoU'],
                                             evalIoU=evalResult['evalIoU'],
                                             evalConfusion=evalResult['evalConfusion'])
                    print('Train Confusion:')
                    trainEvaluator.traverseConfusion()
                    print('Eval Confusion:')
                    evalResult['evalEvaluator'].traverseConfusion()
                    time.sleep(0.1)

    def validation(self,
                   valLoader: MyDataset,  # Dataset,
                   valData: DataLoader,
                   isSave: bool,
                   savePath: Optional[str] = None,
                   epoch: Optional[int] = None) -> Dict:
        evalEvaluator = Evaluator(numClasses=self.args.numClasses)
        if self.args.isTest:
            seed_torch()
            action = 'Test'
        else:
            action = 'Eval'
        tbar = tqdm(valData)
        evalLoss = 0

        idx = 0
        net = self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                oriImg = None
                if len(sample) == 3:
                    oriImg = sample['oriImg']
                    oriImg = oriImg.numpy()
                    oriImg = [j for j in oriImg]

                valImg = sample['img'].to(self.device)
                valLabel = sample['label'].to(self.device)

                out = net(valImg)
                loss = self.segLoss(out, valLabel, mode=self.args.lossType)

                evalLoss += loss.item()
                tbar.set_description('{} Batch:{},{} Loss:{:.5f}'.format(action, i, action, evalLoss / (i + 1)))

                predLabel = out.max(dim=1)[1].data.cpu().numpy()
                predLabel = [j for j in predLabel]

                trueLabel = valLabel.data.cpu().numpy()
                trueLabel = [j for j in trueLabel]

                evalEvaluator.calcSemanticSegmentationConfusion(predLabels=predLabel, gtLabels=trueLabel)

                if self.args.visualizeImage:
                    if self.args.isTest:
                        gridImage = self._visualizeImage(valLoader=valLoader,
                                                         predLabel=predLabel,
                                                         isSave=isSave,
                                                         sample=sample,
                                                         idx=idx,
                                                         savePath=savePath,
                                                         oriImg=oriImg,
                                                         trueLabel=trueLabel,
                                                         i=i)
                        idx += sample['img'].shape[0]
                    else:
                        if i < 3:
                            gridImage = self._visualizeImage(valLoader=valLoader,
                                                             predLabel=predLabel,
                                                             isSave=isSave,
                                                             sample=sample,
                                                             idx=idx,
                                                             savePath=savePath,
                                                             oriImg=oriImg,
                                                             trueLabel=trueLabel,
                                                             i=i)
                            idx += sample['img'].shape[0]
        evalAcc = evalEvaluator.calcSemanticSegmentationAcc()
        evalPixelAcc = evalAcc['pixelAcc']
        evalMeanClassAcc = evalAcc['meanClassAcc']
        evalClassAcc = evalAcc['classAcc']
        evalIoUs = evalEvaluator.calcSemanticSegmentationIoU()
        evalMIoU = evalIoUs['miou']
        evalFWIoU = evalIoUs['fwiou']
        evalIoU = evalIoUs['iou']

        if not self.args.isTest:
            print()
            print('|Epoch|:{}'.format(epoch))
            print('|Eval Loss|:{:.5f}'.format(evalLoss / len(valData)))
            print('|Eval Pixel Acc|:{:.5f}'.format(evalPixelAcc))
            print('|Eval Mean Class Acc|:{:.5f}'.format(evalMeanClassAcc))
            print('|Eval Mean IoU|:{:.5f}'.format(evalMIoU))
            print('|Eval FWIoU|:{:.5f}'.format(evalFWIoU))
            for className, acc, iou in zip(self.args.classNames, evalClassAcc, evalIoU):
                print('|Eval ' + className + ' Accuracy|:{:.5f}'.format(acc))
                print('|Eval ' + className + ' IoU|:{:.5f}'.format(iou))

            newPred = evalMIoU
            if newPred > self.bestPred:
                isBest = True
                self.bestPred = newPred
            else:
                isBest = False
            self.saver.saveCheckpoint({
                'epoch': epoch + 1,
                'stateDict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'bestPred': self.bestPred,
            }, isBest)

        else:
            print()
            print('|Test Loss|:{:.5f}'.format(evalLoss / len(valData)))
            print('|Test Pixel Acc|:{:.5f}'.format(evalPixelAcc))
            print('|Test Mean Class Acc|:{:.5f}'.format(evalMeanClassAcc))
            print('|Test Mean IoU|:{:.5f}'.format(evalMIoU))
            print('|Test FWIoU|:{:.5f}'.format(evalFWIoU))
            for className, acc, iou in zip(self.args.classNames, evalClassAcc, evalIoU):
                print('|Test ' + className + ' Accuracy|:{:.5f}'.format(acc))
                print('|Test ' + className + ' IoU|:{:.5f}'.format(iou))

            self.saver.saveIteration(trainLoss=evalLoss / len(valData),
                                     trainPixelAcc=evalPixelAcc,
                                     trainMeanClassAcc=evalMeanClassAcc,
                                     trainClassAcc=evalClassAcc,
                                     trainMIoU=evalMIoU,
                                     trainFWIoU=evalFWIoU,
                                     trainIoU=evalIoU,
                                     trainConfusion=evalEvaluator.confusion)
            evalEvaluator.traverseConfusion()
        return {'evalLoss': evalLoss / len(valData),
                'evalPixelAcc': evalPixelAcc,
                'evalMeanClassAcc': evalMeanClassAcc,
                'evalClassAcc': evalClassAcc,
                'evalMIoU': evalMIoU,
                'evalFWIoU': evalFWIoU,
                'evalIoU': evalIoU,
                'evalConfusion': evalEvaluator.confusion,
                'evalEvaluator': evalEvaluator}

    def _visualizeImage(self,
                        valLoader: MyDataset,  # Dataset,
                        predLabel: list,
                        isSave: bool,
                        sample: dict,
                        idx: int,
                        savePath: str,
                        oriImg: list,
                        trueLabel: list,
                        i: int):
        preds = valLoader.lp.decodeLabelImages(mode='decodeType1',
                                               encodeLabels=predLabel,
                                               numClasses=self.args.numClasses,
                                               isSave=isSave,
                                               filePaths=valLoader.images[idx:idx + sample['img'].shape[0]],
                                               savePath=savePath,
                                               classNames=self.args.classNames,
                                               originImages=oriImg)
        gts = valLoader.lp.decodeLabelImages(mode='decodeType1', encodeLabels=trueLabel,
                                             numClasses=self.args.numClasses, isSave=False)
        gridImage = plotGridImages(nrow=sample['img'].shape[0],
                                   isSave=isSave,
                                   preds=preds,
                                   labels=gts,
                                   images=oriImg,
                                   savePath=savePath,
                                   idx=i)
        return gridImage

    def run(self):
        if self.args.isTest:
            self.validation(valLoader=self.testLoader,
                            valData=self.testData,
                            isSave=True,
                            savePath=os.path.join(self.saver.experimentDir, 'pred'),
                            epoch=None)
        else:
            print('Starting Epoch:', self.args.startEpoch)
            print('Total Epochs:', self.args.epochs)
            time.sleep(0.1)
            self.training(self.trainData)
