python3 ./script_folder/train_hal.py \
    --n_way $1 \
    --n_shot $2 \
    --n_aug $3 \
    --n_query_all $4 \
    --fc_dim $5 \
    --n_train_class $6 \
    --extractor_folder $7 \
    --num_epoch $8 \
    --result_path . \
    --hallucinator_name HAL_PN_DFHN_1_20_02_02_20_g01_m$1n$2a$3q$4_ep$8 \
    --DFHN \
    --lambda_meta 1.0 \
    --lambda_recon 20.0 \
    --lambda_consistency 0.2 \
    --lambda_consistency_pose 0.2 \
    --lambda_intra 20.0 \
    --lambda_gan 0.1
