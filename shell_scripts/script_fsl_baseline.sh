for lr_power in 6 5 4 3 2
do
    for lr_base in 1 3
    do
        for ite_idx in 0 1 2 3 4 
        do
            python3 ./script_folder/train_fsl.py \
                --n_shot $1 \
                --n_aug $2 \
                --fc_dim $3 \
                --n_class $4 \
                --n_base_class $5 \
                --extractor_folder $6 \
                --exp_tag $7 \
                --hallucinator_name baseline \
                --learning_rate ${lr_base}e-${lr_power} \
                --ite_idx ${ite_idx} \
                --result_path . \
                --model_name FSL_PN_shot$1aug$2_lr${lr_base}e${lr_power}_ite${ite_idx}
        done
        rm -rf $hallucinator_path/FSL*
    done
done
