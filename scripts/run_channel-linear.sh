export CUDA_VISIBLE_DEVICES=2



for dataset in ETTm2 traffic exchange_rate weather electricity
do 
    for pred_len in 96 192 336 720
    do
        python Evolution.py \
            --model_type=Transformer_ChannelTokens_noSeg \
            --dataset=$dataset --pred_len=$pred_len \
            --features=M --num_workers=4 \
            --train_epochs=50 \
            --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
            --d_model=0 \
            --dec_name=linear --dec_self=att --dec_cross=att \
            --embedding=te \
            --ablation_csv_path=exp_settings/final_ChannelTokens_linear_te.csv \
            --save_ckpt \
            --exp_name=final_te
    done
done


for dataset in ETTm2 traffic exchange_rate weather electricity
do 
    for pred_len in 96 192 336 720
    do
        python Evolution.py --model_type=Transformer_ChannelTokens_noSeg \
            --dataset=$dataset --pred_len=$pred_len \
            --features=M --num_workers=4 \
            --train_epochs=50 \
            --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 --d_model=0 \
            --dec_name=linear --dec_self=att --dec_cross=att --embedding=none \
            --ablation_csv_path=exp_settings/final_ChannelTokens_linear.csv \
            --save_ckpt \
            --exp_name=final
    done
done