def get_lr(args):
    if args.model_type == 'Transformer_TimeTokens':
        dict_lr = {
            'ETTm1': 5e-3,
            'ETTm2': 5e-3,
            'ETTh1': 5e-3,
            'ETTh2': 5e-3,
            'electricity': 1e-2,
            'exchange_rate': 1e-2,
            'traffic': 5e-3,
            'weather': 1e-3,
            'national_illness': 1e-2
        }
    elif args.model_type == 'Transformer_ChannelTokens':
        dict_lr = {
            'ETTm1': 1e-2,
            'ETTm2': 1e-2,
            'ETTh1': 1e-2,
            'ETTh2': 1e-2,
            'electricity': 1e-1,
            'exchange_rate': 1e-2,
            'traffic': 1e-1,
            'weather': 1e-3,
            'national_illness': 1e-2
        }
    elif args.model_type == 'Transformer_ChannelTokens_noSeg':
        if args.dec_name=='linear':
            dict_lr = {
                'ETTm1': 5e-2,
                'ETTm2': 5e-2,
                'ETTh1': 5e-2,
                'ETTh2': 5e-2,
                'electricity': 8e-1,
                'exchange_rate': 5e-3,
                'traffic': 5e-1,
                'weather': 5e-2,
                'national_illness': 1e-2
            }
        elif args.dec_name=='transformer':
            dict_lr = {
                'ETTm1': 5e-2,
                'ETTm2': 5e-2,
                'ETTh1': 5e-2,
                'ETTh2': 5e-2,
                'electricity': 8e-1,
                'exchange_rate': 5e-3,
                'traffic': 5e-2,
                'weather': 5e-2,
                'national_illness': 1e-2
            }
    elif args.model_type == 'Transformer_TimeTokens_noSeg':
        dict_lr = {
            'ETTm1': 1e-2,
            'ETTm2': 5e-3,
            'ETTh1': 1e-2,
            'ETTh2': 1e-2,
            'electricity': 1e-2,
            'exchange_rate': 1e-2,
            'traffic': 5e-3,
            'weather': 5e-4,
            'national_illness': 1e-2
        }
    elif args.model_type.lower() == 'tit':
        dict_lr = {
            'ETTm1': 1e-2,
            'ETTm2': 1e-3,
            'ETTh1': 1e-2,
            'ETTh2': 1e-2,
            'electricity': 5e-1,
            'exchange_rate': 5e-2,
            'traffic': 5e-1,
            'weather': 5e-2,
            'national_illness': 1e-2
        }
    elif args.model_type.lower() == 'mlpmixer':
        dict_lr = {
            'ETTm1': 1e-3,
            'ETTm2': 1e-3,
            'ETTh1': 1e-3,
            'ETTh2': 1e-3,
            'electricity': 5e-1,
            'exchange_rate': 5e-2,
            'traffic': 5e-1,
            'weather': 5e-2,
            'national_illness': 5e-2
        }
    else:
        raise Exception('Type {} of model_type for lr is error!'.format(args.model_type))
    
    if args.optimizer == 'sgd':
        learning_rate = dict_lr[args.dataset]
    elif args.optimizer == 'adam':
        learning_rate = 5e-5
    return learning_rate
