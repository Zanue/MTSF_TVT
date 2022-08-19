import torch
import random
import numpy as np
import math
import json

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def get_datainfo(args):
    dict_datapath = {
        'ETTm1': ('./data/ETT-small/', 'ETTm1.csv', 'ETTm1', 96, 48, 7, 7, 7),
        'ETTm2': ('./data/ETT-small/', 'ETTm2.csv', 'ETTm2', 96, 48, 7, 7, 7),
        'ETTh1': ('./data/ETT-small/', 'ETTh1.csv', 'ETTh1', 96, 48, 7, 7, 7),
        'ETTh2': ('./data/ETT-small/', 'ETTh2.csv', 'ETTh2', 96, 48, 7, 7, 7),
        'electricity': ('./data/electricity/', 'electricity.csv', 'custom', 96, 48, 321 ,321 ,321),
        'exchange_rate': ('./data/exchange_rate/', 'exchange_rate.csv', 'custom', 96, 48, 8, 8, 8),
        'traffic': ('./data/traffic/', 'traffic.csv', 'custom', 96, 48, 862, 862, 862),
        'weather': ('./data/weather/', 'weather.csv', 'custom', 96, 48, 21, 21, 21),
        'national_illness': ('./data/illness/', 'national_illness.csv', 'custom', 36, 18, 7, 7, 7)
    }
    # args.root_path, args.data_path, args.data, args.seq_len, args.label_len, args.enc_in, args.dec_in, args.c_out = dict_datapath[args.dataset]
    return dict_datapath[args.dataset]
    
def get_predlen(args):
    if args.model_type == 'Transformer_ChannelTokens':
        beyond_len = 0
        segment_len = args.segment_len
        assert(args.seq_len % segment_len == 0)
        dec_in = args.label_len + args.pred_len
        if dec_in % segment_len == 0:
            return args.pred_len, beyond_len
        else:
            beyond_len = segment_len - (dec_in % segment_len)
            pred_len = beyond_len + args.pred_len
            return pred_len, beyond_len
    else:
        return args.pred_len, 0

def setup_seed(seed = 3407):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.cuda.manual_seed(seed)
     torch.backends.cudnn.benchmark = True

def setup_seed_new(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, saveckpt=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.saveckpt = saveckpt

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.saveckpt:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.saveckpt:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    '''epoch in this function is epoch + 1'''
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj =='type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type5':
        lr_adjust = {epoch: args.learning_rate * (0.97 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type1_warmup':
        start_epoch = 5
        if epoch in range(1, start_epoch):
            lr_adjust = {epoch: 0.2 * start_epoch * args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - start_epoch) // 1))}
    elif args.lradj == 'type5_warmup':
        start_epoch = 5
        if epoch in range(1, start_epoch):
            lr_adjust = {epoch: 0.1 * args.learning_rate * (0.97 ** ((epoch - 1) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate * (0.97 ** ((epoch - start_epoch) // 1))}

    elif args.lradj == 'cosine':
        real_epoch = epoch - 1
        num_epoch = args.train_epochs
        lr_max = args.learning_rate
        old_lr = optimizer.param_groups[0]['lr']
        lr_adjust = {
            epoch: lr_max*0.5*(1 + math.cos(real_epoch/num_epoch*math.pi)) if real_epoch%2==0 else \
                old_lr + 0.5*lr_max*(1 - math.cos(math.pi / num_epoch))
        }
    
    else:
        raise NotImplementedError(f'Learning rate adjustment type = {args.lradj} is not implemented')
        
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

import matplotlib.pyplot as plt  

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
