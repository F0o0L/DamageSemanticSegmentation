import torch.optim.lr_scheduler as lrScheduler
from torch.optim.optimizer import Optimizer
from typing import Optional


class LRScheduler(object):
    def __init__(self,
                 mode: str,
                 optimizer: Optimizer,
                 numEpochs: int,
                 lastEpoch: int = -1) -> None:
        self.mode = mode
        self.optimizer = optimizer
        self.lastEpoch = lastEpoch
        self.numEpochs = numEpochs

        print('Using {} LR Scheduler'.format(self.mode))

    def __call__(self,
                 stepSize: Optional[int] = None,
                 gamma: Optional[float] = None,
                 mileStones: Optional[list] = None):
        scheduler = None
        if self.mode == 'step':
            assert stepSize
            assert gamma
            scheduler = lrScheduler.LambdaLR(optimizer=self.optimizer,
                                             lr_lambda=lambda epoch: gamma ** (epoch // stepSize),
                                             last_epoch=self.lastEpoch)

        elif self.mode == 'multiStep':
            assert mileStones
            assert gamma
            scheduler = lrScheduler.MultiStepLR(optimizer=self.optimizer,
                                                milestones=mileStones,
                                                gamma=gamma,
                                                last_epoch=self.lastEpoch)

        elif self.mode == 'poly':
            scheduler = lrScheduler.LambdaLR(optimizer=self.optimizer,
                                             lr_lambda=lambda epoch: pow((1 - 1.0 * epoch / self.numEpochs), 0.9),
                                             last_epoch=self.lastEpoch)

        elif self.mode == 'reduceLROnPlateau':
            scheduler = lrScheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                      mode='max',
                                                      factor=0.9,
                                                      patience=20)

        return scheduler
