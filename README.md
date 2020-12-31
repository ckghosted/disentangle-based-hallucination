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
- For *N*-way *K*-shot episodic evaluation, use `base.json`, `val.json`, and `novel.json`  
- For multiclass few-shot classification, use `base_train.json`, `base_test.json`, `val_train.json`, `val_test.json`, `novel_train.json`, and `novel_test.json`

## 1. *N*-way *K*-shot Episodic Evaluation
### Step 0
Setup experiment environment
<pre><code>
mkdir -p mini-imagenet/episodic
cd mini-imagenet/episodic
ln -s [path to json files] ./json_folder
ln -s [path to python files] ./py_folder
</code></pre>

### Step 1
Use `train_ext.py` and `model_ext.py` to train a ResNet-18 backbone using the base-class split (`base.json`) and (optionally) extract all base/val/novel features (saved as pickle files: `base_feat`, `val_feat`, and `novel_feat`).
<pre><code>
# Execute the following code under the 'mini-imagenet/episodic' directory.
# The extractor folder (specified by --model_name) will be saved under the 'mini-imagenet' folder.
CUDA_VISIBLE_DEVICES=0 python3 py_folder/train_ext.py \
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

# set link to the extractor folder (execute 'unlink ext1' if necessary)
ln -s ../ResNet18_img224_base_ep100 ./ext1
</code></pre>

### Step 2
Use `train_episodic.py` and `model_episodic.py` to run episodic training (on base classes), validation (on validation classes), and the final evaluation (on novel classes) directly. Please refer to the following examples to execute those `script_PN_XXX.sh` files.
#### baseline
<pre><code>
# arguments:
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
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_PN_baseline_noPro_lr1e5.sh \
    5 1 75 512 64 16 20 3 6 ext1 0 4
</code></pre>

#### PoseRef and AFHN
<pre><code>
# arguments:
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
CUDA_VISIBLE_DEVICES=1 sh ./py_folder/scripts/script_PN_PoseRef_1_1_1_0_0_0_0_g0_tf0_noPro_lr1e5.sh \
    5 1 5 75 512 64 16 20 0 3 0 6 ext1 0 4 10
CUDA_VISIBLE_DEVICES=2 sh ./py_folder/scripts/script_PN_AFHN_1_tf1_ar1_noPro_lr1e5.sh \
    5 1 3 75 512 64 16 20 0 3 0 6 ext1 0 4 10
</code></pre>

## 2. Multiclass Few-shot Classification
### Step 0
Setup experiment environment.
<pre><code>
mkdir -p mini-imagenet/multiclass
cd mini-imagenet/multiclass
ln -s [path to json files] ./json_folder
ln -s [path to python files] ./py_folder
ln -s [path to raw image files] ./image_folder
</code></pre>

### Step 1
Use `train_ext.py` and `model_ext.py` to train a ResNet-18 backbone using the training base-class split (`base_train.json`) and extract all base/val/novel features (saved as pickle files: `base_train_feat`, `base_test_feat`, `val_train_feat`, `val_test_feat`, `novel_train_feat`, and `novel_test_feat`).
<pre><code>
# Execute the following code under the 'mini-imagenet/multiclass' directory.
CUDA_VISIBLE_DEVICES=1 python3 ./py_folder/train_ext.py \
    --result_path . \
    --model_name ResNet18_img224_base_train_ep100_val_centerCrop \
    --n_class 64 \
    --train_path ./json_folder/base_train.json \
    --test_path ./json_folder/base_test.json \
    --num_epoch 100 \
    --bsize 128 \
    --lr_start 1e-3 \
    --lr_decay 0.5 \
    --lr_decay_step 10 \
    --use_aug \
    --with_BN \
    --center_crop \
    --run_extraction > ./log_ResNet18_img224_base_train_ep100_val_centerCrop

# set link to the extractor folder (execute 'unlink ext1' if necessary)
ln -s ./ResNet18_img224_base_train_ep100_val_centerCrop ./ext1
</code></pre>

### Step 2
Use `train_hal.py` and `model_hal.py` to train various hallucinators using the features extracted from `base_train_feat`.
<pre><code>
# arguments:
# $1: n_way
# $2: n_shot
# $3: n_aug
# $4: n_query_all
# $5: z_dim/fc_dim (e.g., 512 for ResNet-18 features)
# $6: n_train_class (number of base categories, e.g., cub: 100; flo: 62; mini-imagenet: 64; cifar-100: 64)
# $7: extractor_folder (ext[1-9] from Step 1, must be specified)
# $8: num_epoch (number of epochs, each containing 600 episodes)
</code></pre>

#### Baseline
<pre><code>
# We do not have to train the hallucinator for the baseline
</code></pre>

#### cGAN
<pre><code>
# lambda_meta=1.0 (no need to specify)
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_cgan.sh 5 1 2 75 512 64 ext1 90 > ./log_hal_cgan_m5n1a2q75_ep90
</code></pre>

#### AFHN
<pre><code>
# lambda_meta=1.0, lambda_tf=1.0, lambda_ar=1.0
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_afhn_1_tf1_ar1.sh 5 1 3 75 512 64 ext1 90 > ./log_hal_afhn_1_tf1_ar1_m5n1a3q75_ep90
</code></pre>

#### DFHN
<pre><code>
# fix lambda_meta=1.0 and lambda_gan=0.1; tune lambda_recon=lambda_intra=X, lambda_consistency=lambda_consistency_pose=0.01*X, where X=1,2,5,10,20,50
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_dfhn_1_2_002_002_2_g01.sh 5 1 2 75 512 64 ext1 90 > ./log_hal_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_dfhn_1_5_005_005_5_g01.sh 5 1 2 75 512 64 ext1 90 > ./log_hal_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_dfhn_1_10_01_01_10_g01.sh 5 1 2 75 512 64 ext1 90 > ./log_hal_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_hal_dfhn_1_20_02_02_20_g01.sh 5 1 2 75 512 64 ext1 90 > ./log_hal_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90
</code></pre>

### Step 3
Use `train_fsl.py` and `model_fsl.py` to sample few shots for each valilation class from `val_train_feat`, augment each class using the trained hallucinator (specified in `script_fsl_XXX.sh`), train a linear classifier using all features in `base_train_feat` and the *augmented* validation features, and finally test on `base_test_feat` and `val_test_feat`. The output can be parsed by `acc_parser.py` and `acc_parser_top1.py`.
<pre><code>
# arguments:
# $1: n_shot (01 or 05)
# $2: n_aug (number of shots per novel category AFTER hallucination, must be equal to n_shot for the baseline)
# $3: z_dim/fc_dim (e.g., 512 for ResNet-18 features)
# $4: n_class (total number of (base + val + novel) categories, e.g., cub: 200; flo: 102; mini-imagenet: 100; cifar-100: 100)
# $5: n_base_class (number of base categories, e.g., cub: 100; flo: 62; mini-imagenet: 64; cifar-100: 64)
# $6: extractor_folder (ext[1-9] from Step 1, must be specified)
# $7: exp_tag ('cv' for imagenet-1k, 'val' for other datasets)
# additional arguments for cGAN, AFHN, and DFHN to specify the parameters used to train the hallucinator
# $8: (m,n,a,q) (e.g., 'm5n1a3q75')
# $9: num_epoch (30 or 60 or 90)
# additional arguments for DFHN on coarse-grained datasets (mini-imagenet or cifar-100)
# $10: n_base_lb_per_novel (number of related base categories for each novel seed feature, default 0: sample reference feature from all base categories)
</code></pre>

#### Baseline
<pre><code>
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_baseline.sh 01 1 512 100 64 ext1 val > ./results_fsl_baseline_shot01_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_baseline_shot01_val > ./results_fsl_baseline_shot01_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_baseline_shot01_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_baseline_shot01_val_acc
</code></pre>

#### cGAN
<pre><code>
# lambda_meta=1.0 (no need to specify)
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_cgan.sh 01 200 512 100 64 ext1 val m5n1a2q75 90 > ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_val > ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_val_acc
</code></pre>

#### AFHN
<pre><code>
# lambda_meta=1.0, lambda_tf=1.0, lambda_ar=1.0
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_afhn_1_tf1_ar1.sh 01 200 512 100 64 ext1 val m5n1a3q75 90 > ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_val > ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_val_acc
</code></pre>

#### DFHN
<pre><code>
# fix lambda_meta=1.0 and lambda_gan=0.1; tune lambda_recon=lambda_intra=X, lambda_consistency=lambda_consistency_pose=0.01*X, where X=1,2,5,10,20,50

CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_dfhn_1_2_002_002_2_g01.sh 01 200 512 100 64 ext1 val m5n1a2q75 90 0 > ./results_fsl_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90_b0_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90_b0_shot01_aug200_val > ./results_fsl_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_dfhn_1_2_002_002_2_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc

CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_dfhn_1_5_005_005_5_g01.sh 01 200 512 100 64 ext1 val m5n1a2q75 90 0 > ./results_fsl_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90_b0_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90_b0_shot01_aug200_val > ./results_fsl_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_dfhn_1_5_005_005_5_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc

CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_dfhn_1_10_01_01_10_g01.sh 01 200 512 100 64 ext1 val m5n1a2q75 90 0 > ./results_fsl_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90_b0_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90_b0_shot01_aug200_val > ./results_fsl_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_dfhn_1_10_01_01_10_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc

CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_dfhn_1_20_02_02_20_g01.sh 01 200 512 100 64 ext1 val m5n1a2q75 90 0 > ./results_fsl_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90_b0_shot01_aug200_val
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90_b0_shot01_aug200_val > ./results_fsl_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser.py ./results_fsl_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_dfhn_1_20_02_02_20_g01_m5n1a2q75_ep90_b0_shot01_aug200_val_acc
</code></pre>

### Step 4
Use the best hyper-parameter setting in **Step 3** to run final evaluation using `base_train_feat`, `novel_train_feat`, `base_test_feat`, and `novel_test_feat`.
<pre><code>
# arguments:
# $1: n_shot (01 or 05)
# $2: n_aug (number of shots per novel category AFTER hallucination, must be equal to n_shot for the baseline)
# $3: z_dim/fc_dim (e.g., 512 for ResNet-18 features)
# $4: n_class (total number of (base + val + novel) categories, e.g., cub: 200; flo: 102; mini-imagenet: 100; cifar-100: 100)
# $5: n_base_class (number of base categories, e.g., cub: 100; flo: 62; mini-imagenet: 64; cifar-100: 64)
# $6: extractor_folder (ext[1-9] from Step 1, must be specified)
# $7: exp_tag ('final' for imagenet-1k, 'novel' for other datasets)
# additional arguments for final evaluation on base + novel categories:
# $8: lr_base (the 'base' part of the best learning rate found through validation, e.g., 3 if the best learning rate is 3e-4)
# $9: lr_power (the negative 'power' part of the best learning rate found through validation, e.g., 4 if the best learning rate is 3e-4)
# additional arguments for cGAN, AFHN, and DFHN to specify the parameters used to train the hallucinator
# $10: (m,n,a,q) (e.g., 'm5n1a3q75')
# $11: num_epoch (30 or 60 or 90)
# additional arguments for DFHN on coarse-grained datasets (mini-imagenet or cifar-100)
# $12: n_base_lb_per_novel (number of related base categories for each novel seed feature, default 0: sample reference feature from all base categories)
</code></pre>

#### Baseline
<pre><code>
# Best learning rate: 3e-5
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_baseline_test.sh 01 1 512 100 64 ext1 novel 3 5 > ./results_fsl_baseline_shot01_novel
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_baseline_shot01_novel > ./results_fsl_baseline_shot01_novel_acc
python3 ./py_folder/acc_parser.py ./results_fsl_baseline_shot01_novel_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_baseline_shot01_novel_acc
</code></pre>

#### cGAN
<pre><code>
# lambda_meta=1.0 (no need to specify)
# Best hallucinator: trained using 5-way 1-shot episodes for 90 epochs
# Best learning rate to train the linear classifier: 3e-4
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_cgan_test.sh 01 200 512 100 64 ext1 novel 3 4 m5n1a2q75 90 > ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_novel
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_novel > ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser.py ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_cgan_m5n1a2q75_ep90_shot01_aug200_novel_acc
</code></pre>

#### AFHN
<pre><code>
# lambda_meta=1.0, lambda_tf=1.0, lambda_ar=1.0
# Best hallucinator: trained using 5-way 1-shot episodes for 90 epochs
# Best learning rate to train the linear classifier: 3e-3
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_afhn_1_tf1_ar1_test.sh 01 200 512 100 64 ext1 novel 3 3 m5n1a3q75 90 > ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_novel
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_novel > ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser.py ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_afhn_1_tf1_ar1_m5n1a3q75_ep90_shot01_aug200_novel_acc
</code></pre>

#### DFHN
<pre><code>
# fix lambda_meta=1.0 and lambda_gan=0.1; tune lambda_recon=lambda_intra=X, lambda_consistency=lambda_consistency_pose=0.01*X, where X=1,2,5,10,20,50
# Best hallucinator: trained using 20-way 1-shot episodes for 90 epochs
# Best learning rate to train the linear classifier: 3e-4
# Best number of related base categories for each novel sample: 20
CUDA_VISIBLE_DEVICES=0 sh ./py_folder/scripts/script_fsl_dfhn_1_10_01_01_10_g01_test.sh 01 200 512 100 64 ext1 novel 3 4 m20n1a2q100 90 20 > ./results_fsl_dfhn_1_10_01_01_10_g01_m20n1a2q100_b20_ep90_shot01_aug200_novel
egrep 'WARNING: the output path|top-5 test accuracy' ./results_fsl_dfhn_1_10_01_01_10_g01_m20n1a2q100_b20_ep90_shot01_aug200_novel > ./results_fsl_dfhn_1_10_01_01_10_g01_m20n1a2q100_b20_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser.py ./results_fsl_dfhn_1_10_01_01_10_g01_m20n1a2q100_b20_ep90_shot01_aug200_novel_acc
python3 ./py_folder/acc_parser_top1.py ./results_fsl_dfhn_1_10_01_01_10_g01_m20n1a2q100_b20_ep90_shot01_aug200_novel_acc
</code></pre>

