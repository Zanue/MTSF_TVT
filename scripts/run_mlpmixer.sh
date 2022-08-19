export CUDA_VISIBLE_DEVICES=2

for dataset in exchange_rate
do
    for pred_len in 96 192 336 720
    do
        python Evolution.py --model_type=MLPMixer --dataset=$dataset \
            --train_epochs=40 --features=M --num_workers=4 \
            --pred_len=$pred_len --batch_size=32 --layer=2 --n_heads=1 --dropout=0.05 \
            --factor=3 --d_model=96 \
            --dec_name=linear --dec_self=att --dec_cross=att --embedding=none \
            --ablation_csv_path=exp_settings/final_mlpmixer.csv \
            --save_ckpt \
            --exp_name=final_mlpmixer
    done
done