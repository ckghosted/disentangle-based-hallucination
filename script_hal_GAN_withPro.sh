for n_way in $1
do
    for n_shot in $2
    do
        for n_aug in $3
        do
            for n_query_all in $4
            do
                for num_epoch in $9
                do
                    for n_ite_per_epoch in 60
                    do
                        python3 ./script_folder/train_hal.py \
                            --result_path . \
                            --extractor_folder $10 \
                            --hallucinator_name HAL_PN_GAN_withPro_m${n_way}n${n_shot}a${n_aug}q${n_query_all}_ep${num_epoch}_$10 \
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
                            --label_key $7 \
                            --exp_tag $8 \
                            --num_parallel_calls $11 \
                            --debug \
                            --GAN \
                            --with_pro
                    done
                done
            done
        done
    done
done