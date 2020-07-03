for n_way in $1
do
    for n_shot in $2
    do
        for n_query_all in $3
        do
            python3 script_folder/train.py \
                --result_path . \
                --json_path json_folder \
                --extractor_path $10 \
                --hallucinator_name HAL_PN_baseline_m${n_way}n${n_shot}q${n_query_all}_ep$8ite$9_$10_$11_noPro_lr1e5 \
                --l2scale 0.0 \
                --num_epoch_pretrain $11 \
                --bsize 128 \
                --lr_start_pre 1e-3 \
                --lr_decay_pre 0.5 \
                --lr_decay_step_pre 10 \
                --num_epoch_noHal $8 \
                --n_ite_per_epoch $9 \
                --lr_start 1e-5 \
                --lr_decay 0.5 \
                --lr_decay_step 20 \
                --patience 20 \
                --n_way ${n_way} \
                --n_shot ${n_shot} \
                --n_query_all ${n_query_all} \
                --z_dim $4 \
                --fc_dim $4 \
                --img_size 224 \
                --c_dim 3 \
                --n_base_class $5 \
                --n_valid_class $6 \
                --n_novel_class $7 \
                --label_key image_labels \
                --lambda_meta 1.0 \
                --lambda_recon 1.0 \
                --lambda_consistency 1.0 \
                --lambda_consistency_pose 0.0 \
                --lambda_intra 0.0 \
                --lambda_pose_code_reg 0.0 \
                --lambda_aux 0.0 \
                --lambda_gan 0.0 \
                --lambda_tf 0.0 \
                --gpu_frac 1.0 \
                --with_BN \
                --num_parallel_calls $12
        done
    done
done
