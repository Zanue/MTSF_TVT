import argparse
import pandas as pd
import torch
from data_provider.data_factory import data_provider
from torch import optim
from models.rank_collapse.TransformerEvolution import Transformer_ChannelTokens_noSeg, Transformer_TimeTokens_noSeg
from models.evolution.MLP import MLPMixer, MLP_ChannelTokens, MLP_TimeTokens
from models.rank_collapse.modules.settings import get_lr
from torch import nn
import time
import numpy as np
from utils.tool import adjust_learning_rate,metric,visual, get_datainfo, EarlyStopping, get_predlen
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.pyplot import plot_seq_feature, plot_seq_correlation
from utils.tool import setup_seed, setup_seed_new
from utils.rank_comp import cal_ratios
from time import sleep
setup_seed_new()



parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--is_training', type=int, default=1, help='status')

parser.add_argument('--device', type=int, default=0, help='gpu dvice')

#method choose
parser.add_argument('--exp_name', type=str, default='baseline', help='Experiment name')
parser.add_argument('--model_type', type=str, default='MLP', help='model type')

# data loader
parser.add_argument('--data', type=str, default='electricity', help='dataset type')
parser.add_argument('--dataset', type=str, default='electricity', help='dataset type', choices=['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', \
    'electricity', 'exchange_rate', 'traffic', 'weather', 'national_illness'])
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--root_path', type=str, default='data/electricity', help='root path of the data file')
parser.add_argument('--save_dir',type=str,default='./ckpt')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                         'S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                         'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--beyond_len', type=int, default=0, help='beyond prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=321, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=321, help='decoder input size')
parser.add_argument('--c_out', type=int, default=321, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--kernel',type=int,default=3,help='kernel size for conv layer')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

#checkpoint_path
parser.add_argument('--check_point',type=str,default='checkpoint',help='check point path')

#hiden layer
parser.add_argument('--hiden',type=int,default=128,help='hiden channel')
parser.add_argument('--layer',type=int,default=1,help='layer of block')
# MLP hidden
parser.add_argument('--middle',type=int,default=128, help='hiden channel')

parser.add_argument('--rnn_layers',type=int,default=1,help='rnn layers num')

# Trend decomp
parser.add_argument('--use_decomp', action='store_true', help='if use trend decomposition')
parser.add_argument('--decomp_ks',type=int,default=25,help='kernel size of avpool')

# supplementary config for FEDformer model
parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
parser.add_argument('--mode_select', type=str, default='random',
                    help='for FEDformer, there are two mode selection method, options: [random, low]')
parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
parser.add_argument('--L', type=int, default=3, help='ignore level')
parser.add_argument('--base', type=str, default='legendre', help='mwt base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='mwt cross atention activation function tanh or softmax')

parser.add_argument('--use_embedding', action='store_true', help='use time stamp embedding for mlpmixer')


# DLinear
parser.add_argument('--individual', action='store_true', help='if each linear is the same', default=False)
# Segment transformer
parser.add_argument('--segment_len', type=int, default=1, help='segment for transformer encoder')
parser.add_argument('--tokenization', type=str, default='none', help='tokenization strategy of transformer')
# channel transformer
parser.add_argument('--n_tokens', type=int, default=-1, help='from channels to n_tokens')
# Layer setting
parser.add_argument('--timelayer', type=str, default='mlp', help='type of time layer')
parser.add_argument('--attentionlayer', type=str, default='att', help='type of attention layer')
parser.add_argument('--decoderlayer', type=str, default='linear', help='type of decoder layer')
# mixformer
parser.add_argument('--learnable_query', action='store_true', help='if query in cross-attention is learnable', default=False)
# embedding
parser.add_argument('--embed_method', type=str, default='add_scale', help='time embedding method')
parser.add_argument('--pre_embed', type=str, default='linear', help='pre-embedding method')
parser.add_argument('--s_scale', action='store_true', help='if scale, only for uni-variate')
parser.add_argument('--fixed_embed', action='store_true', help='if the time embedding is fixed or learnable')
# gradient clip
parser.add_argument('--clip_value', type=float, default=0.1, help='gradient clip')

# weight init
parser.add_argument('--kaiming_normal', action='store_true', help='if use kaiming_normal init')
parser.add_argument('--init_gain', type=float, default=1, help='scale of init weight std')
parser.add_argument('--pe_gain', type=float, default=0.01, help='scale of init pos_embedding std')

# train setting
parser.add_argument('--save_ckpt', action='store_true', help='if save ckpt')
parser.add_argument('--save_fig', action='store_true', help='if save fig of OT variable')

# ablation study .csv file path
parser.add_argument('--ablation_csv_path', type=str, default='./result-ablation_study-zerz.csv')

# decoder ablation
parser.add_argument('--dec_name', type=str, default='transformer')
parser.add_argument('--dec_cross', type=str, default='att')
parser.add_argument('--dec_self', type=str, default='att')
parser.add_argument('--embedding', type=str, default='all', help='embeddings for transformer')

parser.add_argument('--pos_embedding', action='store_true', help='if use pos embedding')


args = parser.parse_args()

# dataset information
args.root_path, args.data_path, args.data, args.seq_len, args.label_len, \
    args.enc_in, args.dec_in, args.c_out = get_datainfo(args)
if args.features == 'S':
    args.enc_in = args.dec_in = args.c_out = 1
args.pred_len, args.beyond_len = get_predlen(args)
# learning rate
args.learning_rate = get_lr(args)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)


if args.model_type == 'MLPMixer':
    model = MLPMixer(args.seq_len,args.label_len, args.pred_len, num_channels=args.enc_in, num_blocks=args.layer, \
        dropout=args.dropout, d_model=args.d_model, embed=args.embed, freq=args.freq)
    expname = '{}/{}/{}/{}_{}_in{}d{}out{}_lr{}_dropout{}_epoch{}' \
        .format(args.data_path[:-4], args.model_type, args.exp_name[:4], \
            args.exp_name[5:], args.features, args.seq_len, args.d_model, args.pred_len, \
                args.learning_rate, args.dropout, args.train_epochs)

elif args.model_type == 'MLP_ChannelTokens':
    model = MLP_ChannelTokens(args.seq_len,args.pred_len, middle=args.middle)
    expname = '{}/{}/{}_{}_in{}out{}_lr{}_bs{}_middle{}_epoch{}' \
        .format(args.data_path[:-4], args.model_type, \
            args.exp_name, args.features, args.seq_len, args.pred_len, \
                args.learning_rate, args.batch_size,args.middle, args.train_epochs)

elif args.model_type == 'MLP_TimeTokens':
    model = MLP_TimeTokens(args.seq_len,args.pred_len, num_channels=args.enc_in, middle=args.middle)
    expname = '{}/{}/{}_{}_in{}out{}_lr{}_bs{}_middle_{}epoch{}' \
        .format(args.data_path[:-4], args.model_type, \
            args.exp_name, args.features, args.seq_len, args.pred_len, \
                args.learning_rate, args.batch_size,args.middle, args.train_epochs)


elif args.model_type == 'Transformer_ChannelTokens_noSeg':
    model = Transformer_ChannelTokens_noSeg(args)
    expname = '{}/{}/{}/{}_Embed-{}_Dec-{}_{}_in{}out{}_lr{}_dropout{}_epoch{}' \
        .format(args.data_path[:-4], args.model_type, args.exp_name[:4], \
            args.exp_name[5:], args.embedding, args.dec_name, \
                args.features, args.seq_len, args.pred_len-args.beyond_len, args.learning_rate, args.dropout, args.train_epochs)

elif args.model_type == 'Transformer_TimeTokens_noSeg':
    model = Transformer_TimeTokens_noSeg(args)
    expname = '{}/{}/{}/{}_Embed-{}_Dec-{}_{}_in{}d{}out{}_lr{}_dropout{}_epoch{}' \
        .format(args.data_path[:-4], args.model_type, args.exp_name[:4], \
            args.exp_name[5:], args.embedding, args.dec_name, \
                args.features, args.seq_len, args.d_model, args.pred_len-args.beyond_len, args.learning_rate, args.dropout, args.train_epochs)

else:
    raise NotImplementedError(f'model type = {args.model_type} is not implemented')

print(model)



device = "cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu"
model.to(device)  
print(expname)


def vali(vali_data, vali_loader, criterion, metric, epoch, writer, flag='vali'):
    total_loss = []
    total_mae = []
    total_ratio = {
        'testratio_pred_tvt': [],
        'testratio_true_tvt': [],
        'testratio_pred_tpt': [],
        'testratio_true_tpt': []
    }
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            if not args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)
            elif flag == 'vali':
                outputs, atts = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)
            else:
                outputs, atts = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, ratio=False) # (B,L,C)
                if i == 0:
                    save_path = os.path.join('./results/atts', expname)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if atts is not None:
                        torch.save(atts, os.path.join(save_path, '{}_TESTatts.pt'.format(epoch)))

            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            if args.beyond_len > 0:
                outputs = outputs[:, :-args.beyond_len, :]
                batch_y = batch_y[:, :-args.beyond_len, :]

            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true).item()
            mae = metric(pred, true).item()
            total_loss.append(loss)
            total_mae.append(mae)
            if flag=='test' and args.output_attention:
                total_ratio['testratio_pred_tvt'].append(cal_ratios(pred.permute(0,2,1)))
                total_ratio['testratio_true_tvt'].append(cal_ratios(true.permute(0,2,1)))
                total_ratio['testratio_pred_tpt'].append(cal_ratios(pred))
                total_ratio['testratio_true_tpt'].append(cal_ratios(true))
                # for key in total_ratio.keys():
                #     total_ratio[key].append(eval(key))

            # if epoch == 1 and args.output_attention and flag == 'test' and i == 0:
            #     torch.save([batch_x.detach(), pred, true], os.path.join(save_path, 'TESTtimeseries.pt'))

            if args.save_fig:
                if i == 0:
                    fig = plot_seq_feature(outputs, batch_y, batch_x, flag)
                    writer.add_figure("figure_{}".format(flag), fig, global_step=epoch)
                    if flag == 'test' and args.output_attention:
                        fig = plot_seq_correlation(outputs.detach(), batch_y.detach(), tokenization='channel')
                        writer.add_figure("figure_test_tvt", fig, global_step=epoch)
                        fig = plot_seq_correlation(outputs.detach(), batch_y.detach(), tokenization='time')
                        writer.add_figure("figure_test_tpt", fig, global_step=epoch)

    total_loss = np.average(total_loss)
    total_mae = np.average(total_mae)
    model.train()

    if flag=='test' and args.output_attention:
        for key in total_ratio.keys():
            tmp = torch.stack(total_ratio[key])
            mean, std = torch.mean(tmp).cpu().numpy(), torch.std(tmp).cpu().numpy()
            total_ratio[key] = np.stack([mean, std])
        return total_loss, total_mae, total_ratio
    else:
        return total_loss, total_mae

def train():
    train_set, train_loader = data_provider(args, "train")
    vali_data, vali_loader = data_provider(args,flag='val')
    test_data, test_loader = data_provider(args,flag='test')
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.learning_rate) 
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(),lr=args.learning_rate) 
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9) 
    else:
        print(f'optimizer = {args.optimizer.lower()} is not implemented, so use SGD')
        optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=0.9) 
    criterion = nn.MSELoss()
    metric = nn.L1Loss()

    train_steps = len(train_loader)

    # early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    writer = SummaryWriter('event/{}'.format(expname))
    # log args setting
    argsDict = args.__dict__
    folder_path = os.path.join('./results', expname)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(folder_path + '/setting.txt','w') as f:
        f.writelines('------------------start--------------------\n')
        for eachArg,value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.write(str(model))
        f.write('\n')
        f.writelines('------------------end--------------------')

    best_vali_loss = np.iinfo(np.int32).max
    best_test_loss = 0
    best_train_loss = 0
    best_epoch = 0
    best_vali_mae = 0
    best_test_mae = 0
    total_test_ratio = {
        'testratio_pred_tvt': [],
        'testratio_true_tvt': [],
        'testratio_pred_tpt': [],
        'testratio_true_tpt': []
    }
    total_train_ratio = {
        'trainratio_pred_tvt': [],
        'trainratio_true_tvt': [],
        'trainratio_pred_tpt': [],
        'trainratio_true_tpt': []
    }

    for epoch in range(args.train_epochs):
        train_loss = []
        train_ratio = {
        'trainratio_pred_tvt': [],
        'trainratio_true_tvt': [],
        'trainratio_pred_tpt': [],
        'trainratio_true_tpt': []
        }
        model.train()
        adjust_learning_rate(optimizer, epoch + 1, args)
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()

            # to cuda
            batch_x = batch_x.float().to(device) # (B,L,C)
            batch_y = batch_y.float().to(device) # (B,L,C)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            if not args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)
            else:
                outputs, atts = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)
                

            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            if args.beyond_len > 0:
                outputs = outputs[:, :-args.beyond_len, :]
                batch_y = batch_y[:, :-args.beyond_len, :]

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            # torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
            optimizer.step()

            if args.output_attention:
                pred, true = outputs.detach(), batch_y.detach()
                train_ratio['trainratio_pred_tvt'].append(cal_ratios(pred.permute(0,2,1)))
                train_ratio['trainratio_true_tvt'].append(cal_ratios(true.permute(0,2,1)))
                train_ratio['trainratio_pred_tpt'].append(cal_ratios(pred))
                train_ratio['trainratio_true_tpt'].append(cal_ratios(true))

            if (i+1) % (train_steps//5) == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))


        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        if args.output_attention:
            for key in train_ratio.keys():
                tmp = torch.stack(train_ratio[key])
                mean, std = torch.mean(tmp).cpu().numpy(), torch.std(tmp).cpu().numpy()
                train_ratio[key] = np.stack([mean, std])

        train_loss = np.average(train_loss)
        vali_loss, vali_mae = vali(vali_data, vali_loader, criterion, metric, epoch, writer, 'vali')
        if not args.output_attention:
            test_loss, test_mae = vali(test_data, test_loader, criterion, metric, epoch, writer, 'test')
        else:
            test_loss, test_mae, test_ratio = vali(test_data, test_loader, criterion, metric, epoch, writer, 'test')
            for key in total_test_ratio.keys():
                total_test_ratio[key].append(test_ratio[key])
            for key in total_train_ratio.keys():
                total_train_ratio[key].append(train_ratio[key])

        print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Vali MAE: {4:.7f} TEST MAE: {5:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss, vali_mae, test_mae))

        # nan -> exit
        if np.isnan(train_loss):
            print('Train loss is nan!!!')
            return 114514, 114514, -1

        if args.save_fig:
            fig = plot_seq_feature(outputs, batch_y, batch_x)
            writer.add_figure("figure_train", fig, global_step=epoch)
            fig = plot_seq_correlation(outputs.detach(), batch_y.detach(), tokenization='channel')
            writer.add_figure("figure_train_tvt", fig, global_step=epoch)
            fig = plot_seq_correlation(outputs.detach(), batch_y.detach(), tokenization='time')
            writer.add_figure("figure_train_tpt", fig, global_step=epoch)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('vali_loss', vali_loss, global_step=epoch)
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        for key in train_ratio.keys():
            writer.add_scalars(key, 
            {
                'mean': train_ratio[key][0], 
                'up_bound': train_ratio[key][0] + train_ratio[key][1],
                'bottom_bound': train_ratio[key][0] - train_ratio[key][1],
            },
            global_step=epoch)
        for key in test_ratio.keys():
            writer.add_scalars(key, 
            {
                'mean': test_ratio[key][0], 
                'up_bound': test_ratio[key][0] + test_ratio[key][1],
                'bottom_bound': test_ratio[key][0] - test_ratio[key][1],
            },
            global_step=epoch)
        
        # early_stopping(vali_loss, model, os.path.join(args.check_point, expname))
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        if vali_loss < best_vali_loss:
            best_vali_loss = vali_loss
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_vali_mae = vali_mae
            best_test_mae = test_mae
            best_epoch = epoch + 1
            if args.save_ckpt:
                ckpt_path = os.path.join(args.check_point, expname)
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                torch.save(model.state_dict(), os.path.join(ckpt_path, 'valid_best_checkpoint.pth'))
        
    if args.save_ckpt:    
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'final_checkpoint.pth'))

    print('Best Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Vali MAE: {4:.7f} Test MAE: {5:.7f}'.format(
        best_epoch, best_train_loss, best_vali_loss, best_test_loss, best_vali_mae, best_test_mae
    ))
    folder_path = './results/exps/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    f = open(folder_path + '/{}_{}_exps.txt'.format(args.exp_name, args.model_type), 'a')
    f.write('\n{} \n'.format(expname))
    f.write('Best Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Vali MAE: {4:.7f} Test MAE: {5:.7f} \n'.format(
        best_epoch, best_train_loss, best_vali_loss, best_test_loss, best_vali_mae, best_test_mae
    ))

    # dict_args = vars(args)
    # dict_args['best_test_loss'] = best_test_loss
    # df = df.append(dict_args, ignore_index=True)
    # df.to_csv(args.ablation_csv_path, index=False)
    ratio_path = os.path.join('./results/atts', expname)
    if not os.path.exists(ratio_path):
        os.makedirs(ratio_path)

    if args.output_attention:
        for key in total_test_ratio.keys():
            total_test_ratio[key] = np.stack(total_test_ratio[key])
        for key in total_train_ratio.keys():
            total_train_ratio[key] = np.stack(total_train_ratio[key])
    np.save(os.path.join(ratio_path, 'TESTratio.npy'), total_test_ratio)
    np.save(os.path.join(ratio_path, 'TRAINratio.npy'), total_train_ratio)

    return best_test_loss,best_test_mae, best_epoch


def test(setting='setting',test=1):
    test_data, test_loader = data_provider(args,flag='test')
    if test:
        print('loading model')
        model.load_state_dict(torch.load(os.path.join(args.check_point, expname, 'valid_best_checkpoint.pth')))

    preds = []
    trues = []
    folder_path = os.path.join('./results', expname, 'visual')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            if not args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)
            else:
                outputs, atts = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # (B,L,C)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            if args.beyond_len > 0:
                outputs = outputs[:, :-args.beyond_len, :]
                batch_y = batch_y[:, :-args.beyond_len, :]

            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = os.path.join('./results', expname, 'metrics')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    f = open(os.path.join('./results', expname, 'result.txt'), 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}'.format(mse, mae))
    f.write('\n')
    f.write('\n')
    f.close()

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    return



#main
best_test_loss, best_test_mae, best_epoch = train()
dict_args = vars(args)
dict_args['best_test_mae'] = best_test_mae
dict_args['best_test_loss'] = best_test_loss
dict_args['best_epoch'] = best_epoch
df = None
if os.path.isfile(args.ablation_csv_path):
    df = pd.read_csv(args.ablation_csv_path)
else:
    df = pd.DataFrame()
df = df.append(dict_args, ignore_index=True)
df.to_csv(args.ablation_csv_path, index=False)
# test()




