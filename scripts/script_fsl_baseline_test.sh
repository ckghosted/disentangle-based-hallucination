for lr_power in $9
do
    for lr_base in $8
    do
        for ite_idx in $(seq 0 1 99);
        do
            python3 ./py_folder/train_fsl.py \
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
