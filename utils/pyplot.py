import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import os
import matplotlib.ticker   


def plot_seq_feature(pred_, true_, origin_=None, label = "train"):
    assert(pred_.shape == true_.shape)

    pred = pred_.detach().clone()[..., -1:]
    true = true_.detach().clone()[..., -1:]
    if origin_ is not None:
        origin = origin_.detach().clone()[..., -1:]

    if len(pred.shape) == 3:  #BLD
        pred = pred[0]
        true = true[0]
        origin = origin[0]
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    origin = origin.cpu().numpy()

    pred = np.concatenate([origin, pred], axis=0)
    true = np.concatenate([origin, true], axis=0)

    L, D = pred.shape
    # if D == 1:
    #     pic_row, pic_col = 1, 1
    # else:
    #     pic_col = 2
    #     pic_row = math.ceil(D/pic_col)
    pic_row, pic_col = D, 1


    fig = plt.figure(figsize=(8*pic_row,8*pic_col))
    for i in range(D):
        ax = plt.subplot(pic_row,pic_col,i+1)
        ax.plot(np.arange(L), pred[:, i], label = "pred")
        ax.plot(np.arange(L), true[:, i], label = "true")
        ax.set_title("dimension = {},  ".format(i) + label)
        ax.legend()

    return fig




def softmax(data):
    data -= data.max(axis=-1)
    return np.exp(data) / np.sum(np.exp(data), axis=-1, keepdims=True)


def mse_similarity(series, tokenization='channel', norm='softmax'): #[L, C]
    if tokenization=='channel':
        series = series.transpose()
    else:
        pass
    N, E = series.shape
    tmp = series.copy()
    mse_sim = np.zeros((N, N))
    for i in range(N):
        mse_sim[i, :] = np.mean((series[i:i+1] - tmp) ** 2, axis=1)
    if norm == 'softmax':
        mse_sim = -softmax(mse_sim)
    elif norm == 'Standardization':
        mse_sim = -(mse_sim - np.mean(mse_sim, axis=1, keepdims=True)) / np.std(mse_sim, axis=1, keepdims=True)
    else:
        mse_sim = -mse_sim
    return mse_sim


def plot_seq_correlation(pred,true, tokenization='channel'):
    '''
    pred, true: [B, L, C]
    '''
    pred = pred.cpu().numpy() #[B, L, C]
    true = true.cpu().numpy()

    sim_pred = mse_similarity(pred[0, :, :], tokenization=tokenization)
    sim_true = mse_similarity(true[0, :, :], tokenization=tokenization)

    cbformat = matplotlib.ticker.ScalarFormatter()   
    cbformat.set_powerlimits((0,0))    
    font_size = 18

    fig=plt.figure(figsize=(12, 6))

    ax=fig.add_subplot(121)
    im=ax.imshow(sim_true,cmap='rainbow')
    ax.tick_params(labelsize=font_size)
    cb = plt.colorbar(im, fraction=0.045, format=cbformat)
    cb.ax.tick_params(labelsize=font_size)
    cb.ax.yaxis.get_offset_text().set_fontsize(font_size)
    plt.title("True", fontdict={'size':30})

    ax=fig.add_subplot(122)
    im=ax.imshow(sim_pred,cmap='rainbow')
    ax.tick_params(labelsize=font_size)
    cb = plt.colorbar(im, fraction=0.045, format=cbformat)
    cb.ax.tick_params(labelsize=font_size)
    cb.ax.yaxis.get_offset_text().set_fontsize(font_size)
    plt.title("Pred", fontdict={'size':30})


    plt.tight_layout()
    return fig