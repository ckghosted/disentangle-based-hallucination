# disentangle-based-hallucination
disentangle-based data hallucination for few-shot learning

# baseline
<pre><code>
# argument:
# $1: n_way
# $2: n_way
# $3: n_query_all
# ---------------
# $4: z_dim/fc_dim
# $5: n_base_class
# $6: n_valid_class
# $7: n_novel_class
# ---------------
# $8: num_epoch_noHal
# $9: n_ite_per_epoch
# ---------------
# $10: extractor_path (ext[1-9] or None: train from scratch)
# $11: num_epoch_pretrain
# $12: num_parallel_calls
CUDA_VISIBLE_DEVICES=0 sh script_folder/script_PN_baseline_noPro_lr1e5.sh 5 1 75 512 64 16 20 3 6 ext3 0 4 > \
    ./log_PN_baseline_m5n1q75_ep3ite6_ext3_0_noPro_lr1e5
</code></pre>

# PoseRef and AFHN
<pre><code>
# argument:
# $1: n_way
# $2: n_way
# $3: n_aug
# $4: n_query_all
# ---------------
# $5: z_dim/fc_dim
# $6: n_base_class
# $7: n_valid_class
# $8: n_novel_class
# ---------------
# $9: num_epoch_noHal
# $10: num_epoch_hal
# $11: num_epoch_joint
# $12: n_ite_per_epoch
# ---------------
# $13: extractor_path (ext[1-9] or None: train from scratch)
# $14: num_epoch_pretrain
# $15: num_parallel_calls
# $16: n_aug_t
CUDA_VISIBLE_DEVICES=0 sh script_folder/script_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_noPro_lr1e5.sh 5 1 5 75 512 64 16 20 1 1 1 6 ext3 0 4 10 > \
    ./log_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_m5n1a5q75_ep1hal1joint1ite6_ext3_0_noPro_lr1e5_testAug10
CUDA_VISIBLE_DEVICES=1 sh script_folder/script_PN_AFHN_1_tf1_ar1_noPro_lr1e5.sh 5 1 5 75 512 64 16 20 1 1 1 6 ext3 0 4 10 > \
    ./log_PN_AFHN_1_tf1_ar1_m5n1a5q75_ep1hal1joint1ite6_ext3_0_noPro_lr1e5_testAug10
</code></pre>