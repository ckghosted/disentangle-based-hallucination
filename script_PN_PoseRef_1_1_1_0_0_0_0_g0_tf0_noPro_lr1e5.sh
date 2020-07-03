for n_way in $1
do
    for n_shot in $2
    do
        for n_aug in $3
        do
            for n_query_all in $4
            do
                python3 script_folder/train.py \
                    --result_path . \
                    --json_path json_folder \
                    --extractor_path $13 \
                    --hallucinator_name HAL_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_m${n_way}n${n_shot}a${n_aug}q${n_query_all}_ep$9hal$10joint$11ite$12_$13_$14_noPro_lr1e5_testAug$16 \
                    --l2scale 0.0 \
                    --num_epoch_pretrain $14 \
                    --bsize 128 \
                    --lr_start_pre 1e-3 \
                    --lr_decay_pre 0.5 \
                    --lr_decay_step_pre 10 \
                    --num_epoch_noHal $9 \
                    --num_epoch_hal $10 \
                    --num_epoch_joint $11 \
                    --n_ite_per_epoch $12 \
                    --lr_start 1e-5 \
                    --lr_decay 0.5 \
                    --lr_decay_step 20 \
                    --patience 20 \
                    --n_way ${n_way} \
                    --n_shot ${n_shot} \
                    --n_aug ${n_aug} \
                    --n_query_all ${n_query_all} \
                    --n_aug_t $16 \
                    --z_dim $5 \
                    --fc_dim $5 \
                    --img_size 224 \
                    --c_dim 3 \
                    --n_base_class $6 \
                    --n_valid_class $7 \
                    --n_novel_class $8 \
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
                    --PoseRef \
                    --with_BN \
                    --num_parallel_calls $15
            done
        done
    done
done
