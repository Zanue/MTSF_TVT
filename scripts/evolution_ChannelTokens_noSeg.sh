export CUDA_VISIBLE_DEVICES=3


for dataset in electricity ETTm2 traffic exchange_rate weather
do 
    for pred_len in 96 192 336 720
    do
        python Evolution.py --model_type=Transformer_ChannelTokens_noSeg \
            --dataset=$dataset --pred_len=$pred_len \
            --features=M --num_workers=10 \
            --train_epochs=40 \
            --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
            --dec_name=transformer --dec_self=att --dec_cross=att --embedding=te \
            --ablation_csv_path=exp_settings/0811_ChannelTokens_noSeg.csv \
            --exp_name=0811_1600_TransformerNoSeg
    done
done



# ili
for pred_len in 24 36 48 60
do
    python Evolution.py --model_type=Transformer_ChannelTokens_noSeg \
        --dataset=national_illness \
        --features=M --num_workers=10 \
        --train_epochs=40 \
        --seq_len=36 --label_len=18 --pred_len=$pred_len \
        --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
        --dec_name=transformer --dec_self=att --dec_cross=att --embedding=te \
        --segment_len=$seg \
        --ablation_csv_path=exp_settings/0811_ChannelTokens_noSeg.csv \
        --exp_name=0811_1600_TransformerNoSeg
done
    