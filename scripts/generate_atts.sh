export CUDA_VISIBLE_DEVICES=0




for dataset in electricity traffic exchange_rate weather ETTm2
do 
    for pred_len in 96 192 336 720
    do
        python lowrank_eval.py --model_type=Transformer_TimeTokens_noSeg \
            --dataset=$dataset --pred_len=$pred_len \
            --features=M --num_workers=4 \
            --train_epochs=20 \
            --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 --d_model=512 \
            --dec_name=transformer --dec_self=att --dec_cross=att --embedding=all \
            --ablation_csv_path=exp_settings/TPT_ratio.csv \
            --output_attention --save_ckpt --save_fig \
            --exp_name=ratio
    done
done
