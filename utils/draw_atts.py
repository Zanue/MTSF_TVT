from genericpath import isdir
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def mkdir(dir):
    if os.path.isdir(dir):
        return
    else:
        os.makedirs(dir)


def plot_atts(name, atts_path, show_range=100, fig_dir='./'):
    atts = torch.load(atts_path)
    enc_atts, dec_self_atts, dec_cross_atts = atts
    enc_att1, enc_att2 = enc_atts
    dec_self_att = dec_self_atts[0]
    dec_cross_atts = dec_cross_atts[0]
    enc_att1 = enc_att1.cpu().numpy()
    enc_att2 = enc_att2.cpu().numpy()
    dec_self_att = dec_self_att.cpu().numpy()
    dec_cross_atts = dec_cross_atts.cpu().numpy()
    print('att shape: enc_att {}, dec_self_att {}, dec_cross_atts {}'.format(enc_att1.shape, dec_self_att.shape, dec_cross_atts.shape))

    enc_att1_map = enc_att1[0,0, :show_range, :show_range]
    enc_att2_map = enc_att2[0,0, :show_range, :show_range]
    dec_self_att_map = dec_self_att[0,0, :show_range, :show_range]
    dec_cross_atts_map = dec_cross_atts[0,0, :show_range, :show_range]


    fig=plt.figure(figsize=(16,16))

    ax1=fig.add_subplot(221)
    im1=ax1.imshow(enc_att1_map,cmap=plt.cm.hot_r)
    plt.colorbar(im1, fraction=0.05)
    plt.title("enc att1")

    ax2=fig.add_subplot(222)
    im2=ax2.imshow(enc_att2_map,cmap=plt.cm.hot_r)
    plt.colorbar(im2, fraction=0.05)
    plt.title("enc att2")

    ax3=fig.add_subplot(223)
    im3=ax3.imshow(dec_self_att_map,cmap=plt.cm.hot_r)
    plt.colorbar(im3, fraction=0.05)
    plt.title("dec self att")

    ax4=fig.add_subplot(224)
    im4=ax4.imshow(dec_cross_atts_map,cmap=plt.cm.hot_r)
    plt.colorbar(im4, fraction=0.05)
    plt.title("dec cross att")

    fig_name = '{}.png'.format(name)
    fig_path = os.path.join(fig_dir,fig_name)

    plt.savefig(fig_path,bbox_inches='tight')
    plt.close()


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


def plot_seq_correlation(name, data_path, show_range=100, tokenization='channel', fig_dir='./'):
    data = torch.load(data_path)
    orgin, pred, true = data
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()

    sim_pred = mse_similarity(pred[0, :, :show_range], tokenization=tokenization)
    sim_true = mse_similarity(true[0, :, :show_range], tokenization=tokenization)

    fig=plt.figure(figsize=(16,16))

    ax1=fig.add_subplot(121)
    im1=ax1.imshow(sim_pred,cmap=plt.cm.hot_r)
    plt.colorbar(im1, fraction=0.05)
    plt.title("Pred")

    ax2=fig.add_subplot(122)
    im2=ax2.imshow(sim_true,cmap=plt.cm.hot_r)
    plt.colorbar(im2, fraction=0.05)
    plt.title("True")


    fig_name = '{}_{}Token.png'.format(name, tokenization)
    fig_path = os.path.join(fig_dir,fig_name)


    plt.savefig(fig_path,bbox_inches='tight')
    plt.close()


def plot_seq(name, data_path, show_range=100, fig_dir='./'):
    data = torch.load(data_path)
    origin, pred, true = data
    pred = pred[0].cpu().numpy()
    true = true[0].cpu().numpy() #[L, C]
    print('seq len: {}'.format(true.shape))

    show_range = min(show_range,pred.shape[-1])

    fig=plt.figure(figsize=(16,16))

    t = np.arange(true.shape[0])
    plt.subplot(211)
    for i in range(show_range):
        plt.plot(t, pred[:, i])
    plt.title("Pred")

    plt.subplot(212)
    for i in range(show_range):
        plt.plot(t, true[:, i])
    plt.title("True")

    fig_name = '{}.png'.format(name)
    fig_path = os.path.join(fig_dir, fig_name)

    plt.savefig(fig_path,bbox_inches='tight')
    plt.close()


def main(
    atts_path = 'results/atts/electricity/Transformer_TimeTokens/0811/2100_generateAtt_Embed-all_Seg1_Dec-transformer_M_in96out96_lr0.1_dropout0.05_epoch40/7_TESTatts.pt',
    data_path = 'results/atts/electricity/Transformer_TimeTokens/0811/2100_generateAtt_Embed-all_Seg1_Dec-transformer_M_in96out96_lr0.1_dropout0.05_epoch40/TESTtimeseries.pt',
    atts_plotname = 'timetoken_att',
    corr_plotname = 'timetoken_seqcor',
    seq_plotname = 'seq',
    tokenization = 'time',
    fig_dir = 'fig/atts'
):  

    assert tokenization in ['time','channel']
    plot_atts(atts_plotname, atts_path, show_range=150, fig_dir=fig_dir)
    plot_seq_correlation(corr_plotname, data_path, show_range=100, tokenization=tokenization, fig_dir=fig_dir)
    plot_seq(seq_plotname, data_path, show_range=100, fig_dir=fig_dir)


if __name__ == "__main__":
    main()