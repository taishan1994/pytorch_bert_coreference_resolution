import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,
                 monitor,
                 patience=7,
                 verbose=False,
                 mode='min',
                 path='checkpoint.pt',
                 trace_func=print
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        assert mode in ['min','max']
        self.monitor = monitor,
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.init_score = None
        self.early_stop = False
        self.init_loss = None
        self.path = path
        self.trace_func = trace_func

    def __call__(self, value, model):
        """
        这里简单认为：
        损失如果在一定时间内没有下降，则进行早停（最小化的参数也行）
        评价指标如果在一定时间内没有上升，则进行早停（最大化的参数也行）
        :param val_loss:
        :param model:
        :return:
        """
        if self.mode == 'min': # 对损失进行早停
            if self.init_loss is None:
                self.init_loss = value
                # self.save_checkpoint(val_loss, model)
            elif value > self.init_loss:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.init_loss = value
                # self.save_checkpoint(value, model)
                self.counter = 0
        elif self.mode == 'max': # 对评价指标进行早停
            if self.init_score is None:
                self.init_score = value
                # self.save_checkpoint(val_loss, model)
            elif value < self.init_score:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.init_score = value
                # self.save_checkpoint(value, model)
                self.counter = 0

    def save_checkpoint(self, value, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.monitor:.6f} --> {value:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.monitor = value
