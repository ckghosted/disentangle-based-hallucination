for n_way in $1
do
    for n_shot in $2
    do
        for n_aug in $3
        do
            for n_query_all in $4
            do
                python3 script_folder/train_episodic.py \
                    --result_path . \
                    --json_path json_folder \
                    --extractor_folder $13 \
                    --hallucinator_name HAL_PN_AFHN_1_tf1_ar1_m${n_way}n${n_shot}a${n_aug}q${n_query_all}_ep$9hal$10joint$11ite$12_$13_$14_noPro_lr1e5_testAug$15 \
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
                    --n_aug_t $15 \
                    --z_dim $5 \
                    --fc_dim $5 \
                    --n_base_class $6 \
                    --n_valid_class $7 \
                    --n_novel_class $8 \
                    --lambda_meta 1.0 \
                    --lambda_tf 1.0 \
                    --lambda_ar 1.0 \
                    --gpu_frac 1.0 \
                    --AFHN \
                    --with_BN \
                    --num_parallel_calls $16
            done
        done
    done
done
