HAL_FOLDERS=./HAL_PN_GAN_m*ext2

for hallucinator_path in $HAL_FOLDERS
do
    echo "hallucinator_path: $hallucinator_path"
    for lr_power in 5
    do
        for lr_base in  1
        do
            for n_shot in $1
            do
                for n_aug in $2
                do
                    for ite_idx in 0
                    do
                        python3 ./script_folder/train_fsl.py \
                            --result_path . \
                            --model_name FSL_PN_shot${n_shot}_GAN_lr${lr_base}e${lr_power}_ite${num_ite}_${ite_idx} \
                            --extractor_folder $7 \
                            --hallucinator_name baseline \
                            --n_class $4 \
                            --n_base_class $5 \
                            --n_shot $((n_shot)) \
                            --n_aug ${n_aug} \
                            --n_top 5 \
                            --bsize $10 \
                            --learning_rate ${lr_base}e-${lr_power} \
                            --l2scale 0.0 \
                            --num_ite $9 \
                            --z_dim $3 \
                            --fc_dim $3 \
                            --label_key $6 \
                            --exp_tag $8 \
                            --gpu_frac 1.0 \
                            --with_BN \
                            --ite_idx ${ite_idx} \
                            --GAN \
                            --debug
                    done
                    # rm -rf $hallucinator_path/FSL*
                done
            done
        done
    done
done
