import torch
import argparse
from Train import Trainer


def getArgparser():
    parser = argparse.ArgumentParser('MyModel')

    # choose model
    parser.add_argument('--model',
                        type=str, default='DeeplabV3Plus',
                        choices=['DeeplabV3Plus', 'FCN'])

    # parameters of Deeplab V3+
    parser.add_argument('--backbone',
                        type=str,
                        default='Xception',
                        choices=['Xception', 'MobileNetV3Small', 'MobileNetV3Large', 'MobileNetV2', 'DRN'])
    parser.add_argument('--ASPP',
                        type=str,
                        default='ASPP',
                        choices=['ASPP', 'LRASPP'])
    parser.add_argument('--outputStride',
                        type=int,
                        default=16,
                        choices=[8, 16, 32])
    parser.add_argument('--lowLevelOutputStride',
                        type=int,
                        default=8,
                        choices=[4, 8, 16])
    parser.add_argument('--preTrained',
                        action='store_true',
                        default=False)
    parser.add_argument('--dilated',
                        action='store_true',
                        default=False)
    parser.add_argument('--modify',
                        action='store_true',
                        default=False)
    parser.add_argument('--dilations',
                        type=str,
                        default='[1,3,6,9]')

    # define dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='CamVid',
                        choices=['CamVid', 'CrackAndCorrosion'])
    parser.add_argument('--baseSize',
                        type=int,
                        default=360)
    parser.add_argument('--cropSize',
                        type=int,
                        default=360)
    parser.add_argument('--testCropSize',
                        type=int,
                        default=360)

    # train parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=50)
    parser.add_argument('--startEpoch',
                        type=int,
                        default=0,
                        help='start epochs (default:0)')
    parser.add_argument('--batchSize',
                        type=int,
                        default=4)
    parser.add_argument('--testBatchSize',
                        type=int,
                        default=4)
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001)
    parser.add_argument('--lrScheduler',
                        type=str,
                        default='step',
                        choices=['step', 'multiStep', 'poly', 'reduceLROnPlateau'])
    parser.add_argument('--lossType',
                        type=str,
                        default='ce',
                        choices=['ce', 'focal', 'dice', 'gdl'])
    parser.add_argument('--useBalancedWeights',
                        action='store_true',
                        default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--noCuda',
                        action='store_true',
                        default=False,
                        help='disable CUDA')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='DataLoader threads')
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--noVal',
                        action='store_true',
                        default=False,
                        help='skip validation during training')
    parser.add_argument('--evalInterval',
                        type=int,
                        default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--isTest',
                        action='store_true',
                        default=False)
    parser.add_argument('--visualizeImage',
                        action='store_true',
                        default=False)
    parser.add_argument('--onlyGtAndPred',
                        action='store_true',
                        default=False)
    parser.add_argument('--noDataAugmentation',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    return args


def getArgparser1():
    parser = argparse.ArgumentParser('MyModel')
    args = parser.parse_args()
    # choose model
    args.model = 'DeeplabV3Plus'  # 'DeeplabV3Plus' 'FCN'

    # parameters of Deeplab V3+
    args.backbone = 'Xception'  # 'Xception' 'MobileNetV3Small' 'MobileNetV3Large' 'MobileNetV2' 'DRN'
    args.ASPP = 'ASPP'  # 'ASPP' 'LRASPP'
    args.outputStride = 16  # 8, 16, 32
    args.lowLevelOutputStride = 4  # 4, 8, 16
    args.preTrained = True
    args.dilated = True
    args.modify = False
    args.dilations = '[1,3,6,9]'  # '[1,6,12,18]'

    # define dataset
    args.dataset = 'CrackAndCorrosion'  # 'CamVid' 'CrackAndCorrosion'
    args.baseSize = 512
    args.cropSize = 512
    args.testCropSize = 512

    # train parameters
    args.epochs = 100
    args.startEpoch = 0
    args.batchSize = 2
    args.testBatchSize = 1
    args.lr = 0.0001
    args.lrScheduler = 'step'  # 'step' 'multiStep' 'poly' 'reduceLROnPlateau'
    args.lossType = 'focal'
    args.useBalancedWeights = False
    args.noCuda = True
    args.workers = 0
    args.resume = None#'F:/PythonWorkspace/temp/Run/CrackAndCorrosion/Deeplab-MobileNetV3Large-ASPP/Experiment0/checkpoint.pth.tar'
    args.noVal = False
    args.evalInterval = 1
    args.isTest = False
    args.visualizeImage = True
    args.onlyGtAndPred = False
    args.noDataAugmentation = True

    return args


def main():
    # args = getArgparser()
    args = getArgparser1()

    args.cuda = not args.noCuda and torch.cuda.is_available()
    args.dilations = eval(args.dilations)

    if 'Deeplab' in str(args.model):
        args.checkName = 'Deeplab-' + str(args.backbone) + '-' + str(args.ASPP)

    elif 'FCN' in str(args.model):
        args.checkName = 'FCN'
    if args.dataset == 'CamVid':
        args.numClasses = 12
        args.classNames = ['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree', 'SignSymbol', 'Fence', 'Car',
                           'Pedestrian', 'Bicyclist', 'unlabelled']
    elif args.dataset == 'CrackAndCorrosion':
        args.numClasses = 3
        args.classNames = ['Corrosion', 'Crack', 'Background']

    argsDict = vars(args)
    for key in argsDict:
        print(key + ':' + str(argsDict[key]))

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
