for n_way in $1
do
    for n_shot in $2
    do
        for n_aug in $3
        do
            for n_query_all in $4
            do
                for num_epoch in $8
                do
                    for n_ite_per_epoch in 600
                    do
                        python3 ./script_folder/train_hal.py \
                            --result_path . \
                            --extractor_folder $9 \
                            --hallucinator_name HAL_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_m${n_way}n${n_shot}a${n_aug}q${n_query_all}_ep${num_epoch}_$9 \
                            --l2scale 0.0 \
                            --n_way ${n_way} \
                            --n_shot ${n_shot} \
                            --n_aug ${n_aug} \
                            --n_query_all ${n_query_all} \
                            --num_epoch ${num_epoch} \
                            --lr_start 1e-5 \
                            --lr_decay 0.5 \
                            --n_ite_per_epoch ${n_ite_per_epoch} \
                            --z_dim $5 \
                            --fc_dim $5 \
                            --n_train_class $6 \
                            --exp_tag $7 \
                            --num_parallel_calls $10 \
                            --debug \
                            --AFHN \
                            --lambda_meta 1.0 \
                            --lambda_recon 1.0 \
                            --lambda_consistency 1.0 \
                            --lambda_consistency_pose 0.0 \
                            --lambda_intra 0.0 \
                            --lambda_pose_code_reg 0.0 \
                            --lambda_aux 0.0 \
                            --lambda_gan 0.0 \
                            --lambda_tf 0.0
                    done
                done
            done
        done
    done
done
