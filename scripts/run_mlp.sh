export CUDA_VISIBLE_DEVICES=3

for dataset in ETTm2 traffic exchange_rate weather electricity
do
    for pred_len in 96 192 336 720
    do
        python Evolution.py --model_type=TiT --dataset=$dataset \
            --train_epochs=40 --features=M --num_workers=4 \
            --pred_len=$pred_len --batch_size=32 --layer=2 --n_heads=1 --dropout=0.05 \
            --factor=3 --d_model=96 \
            --pre_embed=none --attentionlayer=none --timelayer=mlp \
            --dec_name=linear --dec_self=att --dec_cross=att --embedding=none \
            --ablation_csv_path=exp_settings/0815_2100_final_mlp.csv \
            --save_ckpt \
            --exp_name=0815_2100_final_mlp
    done
done