import os
import shutil

import numpy as np
import torch
import glob
from collections import OrderedDict
from argparse import Namespace
from typing import Dict


class Saver(object):
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.directory = os.path.join('Run', args.dataset, args.checkName)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'Experiment*')))
        runId = int(self.runs[-1].split('t')[-1]) + 1 if self.runs else 0
        if args.isTest:
            self.experimentDir = self.directory
        else:
            self.experimentDir = os.path.join(self.directory, 'Experiment{}'.format(str(runId)))
        if not os.path.exists(self.experimentDir):
            os.makedirs(self.experimentDir)

    def saveCheckpoint(self,
                       state: Dict,
                       isBest: bool,
                       filename: str = 'checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experimentDir, filename)
        torch.save(state, filename, _use_new_zipfile_serialization=False)
        if isBest:
            bestPred = state['bestPred']
            with open(os.path.join(self.experimentDir, 'bestPred.txt'), 'w') as f:
                f.write(str(bestPred))
            if not os.path.exists(os.path.join(self.experimentDir, 'best')):
                os.makedirs(os.path.join(self.experimentDir, 'best'))
            shutil.copyfile(filename, os.path.join(os.path.join(self.experimentDir, 'best'), 'experimentBest.pth.tar'))
            if self.runs:
                previousMIOU = [0.0]
                for run in self.runs:
                    runId = run.split('t')[-1]
                    path = os.path.join(self.directory, 'Experiment{}'.format(str(runId)), 'bestPred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            MIOU = float(f.readline())
                            previousMIOU.append(MIOU)
                    else:
                        continue
                maxMIOU = max(previousMIOU)
                if bestPred > maxMIOU:
                    shutil.copyfile(filename, os.path.join(self.directory, 'modelBest.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'modelBest.pth.tar'))

    def saveIteration(self,
                      trainLoss=None,
                      trainPixelAcc=None,
                      trainMeanClassAcc=None,
                      trainClassAcc=None,
                      trainMIoU=None,
                      trainFWIoU=None,
                      trainIoU=None,
                      trainConfusion=None,
                      evalLoss=None,
                      evalPixelAcc=None,
                      evalMeanClassAcc=None,
                      evalClassAcc=None,
                      evalMIoU=None,
                      evalFWIoU=None,
                      evalIoU=None,
                      evalConfusion=None):
        s = ''
        s = s + str(trainLoss) + ','
        s = s + str(trainPixelAcc) + ','
        s = s + str(trainMeanClassAcc)
        for acc in trainClassAcc:
            s = s + ',' + str(acc)
        s = s + ',' + str(trainMIoU)
        s = s + ',' + str(trainFWIoU)
        for iou in trainIoU:
            s = s + ',' + str(iou)
        if evalLoss is not None:
            s = s + ',' + str(evalLoss) + ','
        if evalPixelAcc is not None:
            s = s + str(evalPixelAcc) + ','
        if evalMeanClassAcc is not None:
            s = s + str(evalMeanClassAcc)
        if evalClassAcc is not None:
            for acc in evalClassAcc:
                s = s + ',' + str(acc)
        if evalMIoU is not None:
            s = s + ',' + str(evalMIoU)
        if evalFWIoU is not None:
            s = s + ',' + str(evalFWIoU)
        if evalIoU is not None:
            for iou in evalIoU:
                s = s + ',' + str(iou)
        with open(self.experimentDir + '/iter.txt', "a") as f:
            f.write(s + '\n')

        if evalConfusion is None:
            np.savez(self.experimentDir + '/confusion', trainConfusion)
        else:
            np.savez(self.experimentDir + '/confusion', trainConfusion,evalConfusion)

    def saveExperimentConfig(self):
        logfile = os.path.join(self.experimentDir, 'parameters.txt')
        logFile = open(logfile, 'w')
        p = OrderedDict()
        p['model'] = self.args.model
        if not self.args.model == 'DeeplabV3Plus':
            p['backbone'] = 'None'
            p['ASPP'] = 'None'
            p['outputStride'] = 'None'
            p['lowLevelOutputStride'] = 'None'
            p['preTrained'] = self.args.preTrained
            p['dilated'] = 'None'
            p['modify'] = 'None'
            p['dilations'] = 'None'
        else:
            p['backbone'] = self.args.backbone
            p['ASPP'] = self.args.ASPP
            p['outputStride'] = self.args.outputStride
            if 'MobileNet' in self.args.backbone:
                p['lowLevelOutputStride'] = self.args.lowLevelOutputStride
                p['preTrained'] = self.args.preTrained
                p['dilated'] = self.args.dilated
            else:
                p['lowLevelOutputStride'] = '4'
                p['preTrained'] = self.args.preTrained
                p['dilated'] = 'None'
            if self.args.ASPP == 'ASPP':
                p['modify'] = self.args.modify
                p['dilations'] = self.args.dilations
            else:
                p['modify'] = 'None'
                p['dilations'] = 'None'
        p['dataset'] = self.args.dataset
        p['numClasses'] = self.args.numClasses
        p['classNames'] = self.args.classNames
        p['baseSize'] = self.args.baseSize
        p['cropSize'] = self.args.cropSize
        p['testCropSize'] = self.args.testCropSize

        p['epochs'] = self.args.epochs
        p['startEpoch'] = self.args.startEpoch
        p['batchSize'] = self.args.batchSize
        p['testBatchSize'] = self.args.testBatchSize
        p['lr'] = self.args.lr
        p['lrScheduler'] = self.args.lrScheduler
        p['lossType'] = self.args.lossType
        if self.args.lossType == 'ce':
            p['useBalancedWeights'] = self.args.useBalancedWeights
        else:
            p['useBalancedWeights'] = 'None'
        p['workers'] = self.args.workers
        p['resume'] = self.args.resume
        p['noVal'] = self.args.noVal
        p['evalInterval'] = self.args.evalInterval
        p['isTest'] = self.args.isTest
        p['visualizeImage'] = self.args.visualizeImage
        p['onlyGtAndPred'] = self.args.onlyGtAndPred
        p['noDataAugmentation'] = self.args.noDataAugmentation
        p['cuda'] = self.args.cuda

        for key, val in p.items():
            logFile.write(key + ':' + str(val) + '\n')
        logFile.close()

    def saveRunningTime(self):
        pass