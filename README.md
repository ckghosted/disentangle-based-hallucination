# disentangle-based-hallucination
disentangle-based data hallucination for few-shot learning

## Data Splits
The data splits are provided in the `data-split` folder as `json` files. Please refer to the following python code:
<pre><code>
import json

train_path = './data-split/mini-imagenet/base.json'

with open(train_path, 'r') as reader:
        train_dict = json.loads(reader.read())

train_image_list = train_dict['image_names']
train_class_list = train_dict['image_labels']
</code></pre>
For N-way k-shot episodic evaluation, use `base.json`, `val.json`, and `novel.json`; For few-shot multiclass classification, use `base_train.json`, `base_test.json`, `val_train.json`, `val_test.json`, `novel_train.json`, and `novel_test.json`.

## 1. N-way k-shot Episodic Evaluation
### Step 0
Setup experiment environment
<pre><code>
mkdir -p mini-imagenet/episodic
cd mini-imagenet/episodic
ln -s [path to json files] ./json_folder
ln -s [path to python files] ./script_folder
</code></pre>

### Step 1
Use `train_ext.py` and `model_ext.py` to train a ResNet-18 backbone using the base-class split (`base.json`) and (optionally) extract all base/val/novel features (saved as pickle files: `base_feat`, `val_feat`, and `novel_feat`).
<pre><code>
# (execute the following code under the 'mini-imagenet/episodic' directory)
# (the extractor folder will be saved under the 'mini-imagenet' folder)
CUDA_VISIBLE_DEVICES=0 python3 script_folder/train_ext.py \
    --result_path .. \
    --model_name ResNet18_img224_base_ep100 \
    --n_class 64 \
    --train_path ./json_folder/base.json \
    --num_epoch 100 \
    --bsize 128 \
    --lr_start 1e-3 \
    --lr_decay 0.5 \
    --lr_decay_step 10 \
    --use_aug \
    --with_BN \
    --run_extraction > ../log_ResNet18_img224_base_ep100
# (execute 'unlink ext1' if necessary)
ln -s ../ResNet18_img224_base_ep100 ./ext1
</code></pre>

### Step 2
Use `train_episodic.py` and `model_episodic.py` to run episodic training (on base classes), validation (on validation classes), and the final evaluation (on novel classes) directly. Please refer to the following examples to execute those `script_PN_XXX.sh` files.
#### baseline
<pre><code>
# argument:
# $1: n_way
# $2: n_shot
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
# $10: extractor_folder (ext[1-9] from Step 1, or None: train from scratch)
# $11: num_epoch_pretrain
# ---------------
# $12: num_parallel_calls
CUDA_VISIBLE_DEVICES=0 sh script_folder/script_PN_baseline_noPro_lr1e5.sh 5 1 75 512 64 16 20 3 6 ext1 0 4 > \
    ./log_PN_baseline_m5n1q75_ep3ite6_ext1_0_noPro_lr1e5
</code></pre>

#### PoseRef and AFHN
<pre><code>
# argument:
# $1: n_way
# $2: n_shot
# $3: n_aug (number of samples per class in the augmented support set AFTER hallucination during training)
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
# ---------------
# $15: n_aug_t (number of samples per class in the augmented support set AFTER hallucination during testing)
# ---------------
# $16: num_parallel_calls
CUDA_VISIBLE_DEVICES=1 sh script_folder/script_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_noPro_lr1e5.sh 5 1 5 75 512 64 16 20 0 3 0 6 ext1 0 4 10 > \
    ./log_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_m5n1a5q75_ep0hal3joint0ite6_ext1_0_noPro_lr1e5_testAug10
CUDA_VISIBLE_DEVICES=2 sh script_folder/script_PN_AFHN_1_tf1_ar1_noPro_lr1e5.sh 5 1 3 75 512 64 16 20 0 3 0 6 ext1 0 4 10 > \
    ./log_PN_AFHN_1_tf1_ar1_m5n1a3q75_ep0hal3joint0ite6_ext1_0_noPro_lr1e5_testAug10
</code></pre>

## 2. Few-shot Multiclass Classification
### Step 0
Setup experiment environment
<pre><code>
mkdir -p mini-imagenet/multiclass
cd mini-imagenet/multiclass
ln -s [path to json files] ./json_folder
ln -s [path to python files] ./script_folder
</code></pre>

### Step 1
Use `train_ext.py` and `model_ext.py` to train a ResNet-18 backbone using the training base-class split (`base_train.json`) and extract all base/val/novel features (saved as pickle files: `base_train_feat`, `base_test_feat`, `val_train_feat`, `val_test_feat`, `novel_train_feat`, and `novel_test_feat`).
<pre><code>
# (execute the following code under the 'mini-imagenet/multiclass' directory)
# (the extractor folder will be saved under the 'mini-imagenet' folder)
CUDA_VISIBLE_DEVICES=1 python3 ./script_folder/train_ext.py \
    --result_path .. \
    --model_name ResNet18_img224_base_train_ep100 \
    --n_class 64 \
    --train_path ./json_folder/base_train.json \
    --test_path None \
    --num_epoch 100 \
    --bsize 128 \
    --lr_start 1e-3 \
    --lr_decay 0.5 \
    --lr_decay_step 10 \
    --use_aug \
    --with_BN \
    --run_extraction > ../log_ResNet18_img224_base_train_ep100
# (execute 'unlink ext1' if necessary)
ln -s ../ResNet18_img224_base_train_ep100 ./ext1
</code></pre>

### Step 2
Use `train_hal.py` and `model_hal.py` to train various hallucinators using the features extracted from the training base-class split (`base_train.json`).
<pre><code>
# argument:
# $1: n_way
# $2: n_shot
# $3: n_aug
# $4: n_query_all
# ----------------
# $5: z_dim/fc_dim
# $6: n_train_class
# $7: exp_tag (cv/final for imagenet-1k, common for other datasets)
# ----------------
# $8: num_epoch
# $9: extractor_folder (ext[1-9] from Step 1, must be specified)
CUDA_VISIBLE_DEVICES=2 sh script_folder/script_hal_GAN_withPro.sh 5 1 3 20 512 64 common 100 ext1 > ./log_hal_GAN_withPro_m5n1a3q20_ep100_ext1
CUDA_VISIBLE_DEVICES=3 sh script_folder/script_hal_AFHN_1_tf1_ar1_noPro.sh 5 1 3 20 512 64 common 100 ext1 > ./log_hal_AFHN_1_tf1_ar1_noPro_m5n1a3q20_ep100_ext1
CUDA_VISIBLE_DEVICES=4 sh script_folder/script_hal_PoseRef_1_1_1_0_0_0_0_g0_tf0_withPro.sh 5 1 3 20 512 64 common 100 ext1 > ./log_hal_PoseRef_1_1_1_0_0_0_0_g0_tf0_withPro_m5n1a3q20_ep100_ext1
</code></pre>

### Step 3
Use `train_fsl.py` and `model_fsl.py` to sample few shots for each valilation class (or each novel class), augment each class using the trained hallucinator (specified in script_fsl_XXX.sh), train a linear classifier using all training base-class features and the augmented validation (or novel) features, and finally test on the features extracted from the test splits (i.e., `base_test.json`, `val_test.json`, and `novel_test.json`).
<pre><code>
# argument:
# $1: n_shot (01, 02, 05, 10, or 20)
# $2: n_aug (must be equal to n_shot for baseline)
# $3: z_dim/fc_dim
# $4: n_class
# $5: n_base_class
# $6: extractor_folder (ext[1-9] from Step 1, must be specified)
# $7: exp_tag (cv/final for imagenet-1k, val/novel for other datasets)
# $8: num_ite (10000 for imagenet-1k, 2000 for other datasets)
# $9: bsize (1000 for imagenet-1k, 200 for other datasets)
CUDA_VISIBLE_DEVICES=1 sh script_folder/script_fsl_baseline.sh 01 1 512 80 64 ext1 val 2000 200 > ./results_baseline_val_shot01_ext1
CUDA_VISIBLE_DEVICES=1 sh script_folder/script_fsl_GAN.sh 01 5 512 80 64 ext1 val 2000 200 > ./results_GAN_a5_val_shot01_ext1
</code></pre>

### Collect results
The output files in Step 3 can be parsed by `acc_parser.py` as follows:
<pre><code>
egrep 'WARNING: the output path|top-5 test accuracy' ./results_baseline_val_shot01_ext1 > ./results_baseline_val_shot01_ext1_acc
python3 script_folder/acc_parser.py ./results_baseline_val_shot01_ext1_acc
</code></pre>
Each row contains the used learning rate, n_shot, n_aug, the averaged novel-class accuracy and its std, and the averaged all-class accuracy and its std.

