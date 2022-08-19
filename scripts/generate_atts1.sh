export CUDA_VISIBLE_DEVICES=1



for dataset in electricity traffic ETTm2 exchange_rate weather 
do 
    for pred_len in 96 192 336 720
    do
        python lowrank_eval.py --model_type=Transformer_ChannelTokens_noSeg \
            --dataset=$dataset --pred_len=$pred_len \
            --features=M --num_workers=4 \
            --train_epochs=20 \
            --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 --d_model=0 \
            --dec_name=linear --dec_self=att --dec_cross=att --embedding=none \
            --ablation_csv_path=exp_settings/TVT_ratio.csv \
            --output_attention --save_ckpt --save_fig \
            --exp_name=ratio
    done
done
