export CUDA_VISIBLE_DEVICES=1




# for dataset in ETTm2 traffic exchange_rate electricity
# do 
#     for pred_len in 96
#     do
#         python lowrank_eval.py --model_type=Transformer_TimeTokens \
#             --dataset=$dataset --pred_len=$pred_len \
#             --features=M --num_workers=4 \
#             --train_epochs=30 \
#             --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
#             --dec_name=transformer --dec_self=att --dec_cross=att --embedding=all \
#             --segment_len=1 \
#             --ablation_csv_path=exp_settings/tmp.csv \
#             --output_attention \
#             --exp_name=0816_1500_Att-ratio
#     done
# done


# for dataset in ETTm2 traffic exchange_rate weather electricity
# do 
#     for pred_len in 96
#     do
#         python lowrank_eval.py --model_type=Transformer_ChannelTokens_noSeg \
#             --dataset=$dataset --pred_len=$pred_len \
#             --features=M --num_workers=10 \
#             --train_epochs=40 \
#             --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
#             --dec_name=transformer --dec_self=att --dec_cross=att --embedding=te \
#             --ablation_csv_path=exp_settings/tmp.csv \
#             --output_attention \
#             --exp_name=0812_0800_Att-ratio
#     done
# done


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
            --ablation_csv_path=exp_settings/0818_2100_TVT_ratio.csv \
            --output_attention --save_ckpt --save_fig \
            --exp_name=0818_2100_ratio
    done
done


# for dataset in electricity
# do 
#     for pred_len in 96
#     do
#         python Evolution.py --model_type=Transformer_ChannelTokens_noSeg \
#             --dataset=$dataset --pred_len=$pred_len \
#             --features=M --num_workers=10 \
#             --train_epochs=40 \
#             --batch_size=32 --e_layers=2 --d_layers=1 --n_heads=1 --dropout=0.05 \
#             --dec_name=transformer --dec_self=att --dec_cross=att --embedding=te \
#             --ablation_csv_path=exp_settings/tmp.csv \
#             --output_attention \
#             --exp_name=0811_1900_generateAtt
#     done
# done