export CUDA_VISIBLE_DEVICES=1

for dataset in ETTm2 traffic exchange_rate weather electricity
do
    for pred_len in 96 192 336 720
    do
        python Evolution.py --model_type=TiT --dataset=$dataset \
            --train_epochs=40 --features=M --num_workers=4 \
            --pred_len=$pred_len --batch_size=32 --layer=2 --n_heads=1 --dropout=0.05 \
            --factor=3 --d_model=96 \
            --pre_embed=none --attentionlayer=att --timelayer=mlp \
            --dec_name=linear --dec_self=att --dec_cross=att --embedding=none \
            --ablation_csv_path=exp_settings/tmp.csv \
            --pos_embedding  --pe_gain=0.01 \
            --exp_name=0813_2100_TiTPos
    done
done