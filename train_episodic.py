import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_episodic import HAL_PN_baseline, HAL_PN_GAN, HAL_PN_GAN2, HAL_PN_AFHN, HAL_PN_PoseRef
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--json_path', type=str, help='Folder name containing base.json, val.json, and novel.json')
    parser.add_argument('--extractor_folder', type=str, help='Folder name of the pre-trained feature extractor model')
    parser.add_argument('--hallucinator_name', type=str, help='Folder name to save hallucinator models and learning curves')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--n_way', default=40, type=int, help='Number of classes in the support set')
    parser.add_argument('--n_shot', default=5, type=int, help='Number of samples per class in the support set')
    parser.add_argument('--n_aug', default=20, type=int, help='Number of samples per class in the augmented support set')
    parser.add_argument('--n_query_all', default=100, type=int, help='Number of samples in the query set')
    parser.add_argument('--n_way_t', default=5, type=int, help='Number of classes in the support set for testing')
    parser.add_argument('--n_shot_t', default=1, type=int, help='Number of samples per class in the support set for testing')
    parser.add_argument('--n_aug_t', default=5, type=int, help='Number of samples per class in the augmented support set for testing')
    parser.add_argument('--n_query_all_t', default=75, type=int, help='Number of samples in the query set for testing')
    parser.add_argument('--num_epoch_pretrain', default=100, type=int, help='Number of epochs to pre-train the feature extractor')
    parser.add_argument('--num_epoch_noHal', default=0, type=int, help='Number of epochs to train the prototypical network without hallucination')
    parser.add_argument('--num_epoch_hal', default=0, type=int, help='Number of epochs to train the hallucinator with feature extractor fixed')
    parser.add_argument('--num_epoch_joint', default=0, type=int, help='Number of epochs to train the whole model')
    parser.add_argument('--bsize', default=128, type=int, help='Batch size to pre-train the feature extractor')
    parser.add_argument('--lr_start_pre', default=1e-3, type=float, help='Initial learning rate for pre-training the feature extractor')
    parser.add_argument('--lr_decay_pre', default=0.5, type=float, help='Learning rate decay factor for pre-training the feature extractor')
    parser.add_argument('--lr_decay_step_pre', default=10, type=int, help='Number of epochs per learning rate decay for pre-training the feature extractor')
    parser.add_argument('--lr_start', default=1e-5, type=float, help='Initial learning rate for episodic training')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='Learning rate decay factor for episodic training')
    parser.add_argument('--lr_decay_step', default=20, type=int, help='Number of epochs per learning rate decay for episodic training')
    parser.add_argument('--patience', default=20, type=int, help='Patience for early-stop mechanism')
    parser.add_argument('--n_ite_per_epoch', default=600, type=int, help='Number of iterations (episodes) per epoch')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--img_size', default=224, type=int, help='Image size H and W')
    parser.add_argument('--c_dim', default=3, type=int, help='Image size C')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
    parser.add_argument('--z_std', default=1.0, type=float, help='Standard deviation of the input noise to the GAN-based hallucinator')
    parser.add_argument('--lambda_meta', default=1.0, type=float, help='lambda_meta')
    parser.add_argument('--lambda_recon', default=1.0, type=float, help='lambda_recon')
    parser.add_argument('--lambda_consistency', default=0.0, type=float, help='lambda_consistency')
    parser.add_argument('--lambda_consistency_pose', default=0.0, type=float, help='lambda_consistency_pose')
    parser.add_argument('--lambda_intra', default=0.0, type=float, help='lambda_intra')
    parser.add_argument('--lambda_pose_code_reg', default=0.0, type=float, help='lambda_pose_code_reg')
    parser.add_argument('--lambda_aux', default=0.0, type=float, help='lambda_aux')
    parser.add_argument('--lambda_gan', default=0.0, type=float, help='lambda_gan')
    parser.add_argument('--lambda_tf', default=0.0, type=float, help='lambda_tf')
    parser.add_argument('--lambda_ar', default=0.0, type=float, help='lambda_ar')
    parser.add_argument('--n_base_class', default=64, type=int, help='Number of base class')
    parser.add_argument('--n_valid_class', default=16, type=int, help='Number of val class')
    parser.add_argument('--n_novel_class', default=20, type=int, help='Number of novel class')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--AFHN', action='store_true', help='Use AFHN if present')
    parser.add_argument('--PoseRef', action='store_true', help='Use pose-ref-based hallucinator if present')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    parser.add_argument('--label_key', default='image_labels', type=str, help='image_labels or image_labels_id')
    parser.add_argument('--gpu_frac', default=0.5, type=float, help='per_process_gpu_memory_fraction (0.0~1.0)')
    parser.add_argument('--with_BN', action='store_true', help='Use batch_norm() in the feature extractor mode if present')
    parser.add_argument('--with_pro', action='store_true', help='Use additional embedding network for prototypical network if present')
    parser.add_argument('--num_parallel_calls', default=8, type=int, help='Number of core used to prepare data')
    args = parser.parse_args()
    train(args)
    inference(args)

# Train the hallucinator
def train(args):
    print('==================== training ====================')
    train_path = os.path.join(args.result_path, args.json_path, 'base.json')
    test_path = os.path.join(args.result_path, args.json_path, 'val.json')
    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.GAN:
            # HAL_PN_GAN: implementation of (Y.-X. Wang, CVPR 2018)
            print('train_hallucinator.py --> main() --> train(): use HAL_PN_GAN')
            net = HAL_PN_GAN(sess,
                             model_name=args.hallucinator_name,
                             result_path=args.result_path,
                             train_path=train_path,
                             test_path=test_path,
                             label_key=args.label_key,
                             bsize=args.bsize,
                             n_way=args.n_way,
                             n_shot=args.n_shot,
                             n_aug=args.n_aug,
                             n_query_all=args.n_query_all,
                             n_way_t=args.n_way_t,
                             n_shot_t=args.n_shot_t,
                             n_aug_t=args.n_aug_t,
                             n_query_all_t=args.n_query_all_t,
                             fc_dim=args.fc_dim,
                             img_size=args.img_size,
                             c_dim=args.c_dim,
                             l2scale=args.l2scale,
                             n_train_class=args.n_base_class,
                             n_test_class=args.n_valid_class,
                             with_BN=args.with_BN,
                             with_pro=args.with_pro,
                             num_parallel_calls=args.num_parallel_calls)
        elif args.GAN2:
            # HAL_PN_GAN2: adds one more layer to the hallucinator of HAL_PN_GAN
            print('train_hallucinator.py --> main() --> train(): use HAL_PN_GAN2')
            net = HAL_PN_GAN2(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              test_path=test_path,
                              label_key=args.label_key,
                              bsize=args.bsize,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              n_way_t=args.n_way_t,
                              n_shot_t=args.n_shot_t,
                              n_aug_t=args.n_aug_t,
                              n_query_all_t=args.n_query_all_t,
                              fc_dim=args.fc_dim,
                              img_size=args.img_size,
                              c_dim=args.c_dim,
                              l2scale=args.l2scale,
                              n_train_class=args.n_base_class,
                              n_test_class=args.n_valid_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls)
        elif args.AFHN:
            print('train_hallucinator.py --> main() --> train(): use HAL_PN_AFHN')
            net = HAL_PN_AFHN(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              test_path=test_path,
                              label_key=args.label_key,
                              bsize=args.bsize,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              n_way_t=args.n_way_t,
                              n_shot_t=args.n_shot_t,
                              n_aug_t=args.n_aug_t,
                              n_query_all_t=args.n_query_all_t,
                              fc_dim=args.fc_dim,
                              img_size=args.img_size,
                              c_dim=args.c_dim,
                              lambda_meta=args.lambda_meta,
                              lambda_tf=args.lambda_tf,
                              lambda_ar=args.lambda_ar,
                              l2scale=args.l2scale,
                              n_train_class=args.n_base_class,
                              n_test_class=args.n_valid_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls)
        elif args.PoseRef:
            print('train_hallucinator.py --> main() --> train(): use HAL_PN_PoseRef')
            net = HAL_PN_PoseRef(sess,
                                 model_name=args.hallucinator_name,
                                 result_path=args.result_path,
                                 train_path=train_path,
                                 test_path=test_path,
                                 label_key=args.label_key,
                                 bsize=args.bsize,
                                 n_way=args.n_way,
                                 n_shot=args.n_shot,
                                 n_aug=args.n_aug,
                                 n_query_all=args.n_query_all,
                                 n_way_t=args.n_way_t,
                                 n_shot_t=args.n_shot_t,
                                 n_aug_t=args.n_aug_t,
                                 n_query_all_t=args.n_query_all_t,
                                 fc_dim=args.fc_dim,
                                 img_size=args.img_size,
                                 c_dim=args.c_dim,
                                 lambda_meta=args.lambda_meta,
                                 lambda_recon=args.lambda_recon,
                                 lambda_consistency=args.lambda_consistency,
                                 lambda_consistency_pose=args.lambda_consistency_pose,
                                 lambda_intra=args.lambda_intra,
                                 lambda_pose_code_reg=args.lambda_pose_code_reg,
                                 lambda_aux=args.lambda_aux,
                                 lambda_gan=args.lambda_gan,
                                 lambda_tf=args.lambda_tf,
                                 l2scale=args.l2scale,
                                 n_train_class=args.n_base_class,
                                 n_test_class=args.n_valid_class,
                                 with_BN=args.with_BN,
                                 with_pro=args.with_pro,
                                 num_parallel_calls=args.num_parallel_calls)
        else:
            print('train_hallucinator.py --> main() --> train(): use HAL_PN_baseline')
            net = HAL_PN_baseline(sess,
                                  model_name=args.hallucinator_name,
                                  result_path=args.result_path,
                                  train_path=train_path,
                                  test_path=test_path,
                                  label_key=args.label_key,
                                  bsize=args.bsize,
                                  n_way=args.n_way,
                                  n_shot=args.n_shot,
                                  n_query_all=args.n_query_all,
                                  n_way_t=args.n_way_t,
                                  n_shot_t=args.n_shot_t,
                                  n_query_all_t=args.n_query_all_t,
                                  fc_dim=args.fc_dim,
                                  img_size=args.img_size,
                                  c_dim=args.c_dim,
                                  l2scale=args.l2scale,
                                  n_train_class=args.n_base_class,
                                  n_test_class=args.n_valid_class,
                                  with_BN=args.with_BN,
                                  with_pro=args.with_pro,
                                  num_parallel_calls=args.num_parallel_calls)
        all_vars, trainable_vars, all_regs = net.build_model()
        # Debug: Check trainable variables and regularizers
        if args.debug:
            print('------------------[all_vars]------------------')
            for var in all_vars:
                print(var.name)
            print('------------------[trainable_vars]------------------')
            for var in trainable_vars:
                print(var.name)
            print('------------------[all_regs]------------------')
            for var in all_regs:
                print(var.name)
        res = net.train(ext_from=os.path.join(args.extractor_folder, 'models'),
                        num_epoch_pretrain=args.num_epoch_pretrain,
                        lr_start_pre=args.lr_start_pre,
                        lr_decay_pre=args.lr_decay_pre,
                        lr_decay_step_pre=args.lr_decay_step_pre,
                        num_epoch_noHal=args.num_epoch_noHal,
                        num_epoch_hal=args.num_epoch_hal,
                        num_epoch_joint=args.num_epoch_joint,
                        n_ite_per_epoch=args.n_ite_per_epoch,
                        lr_start=args.lr_start,
                        lr_decay=args.lr_decay,
                        lr_decay_step=args.lr_decay_step,
                        patience=args.patience)
    np.save(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'), res)
    
    # Plot learning curve
    results = np.load(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'), allow_pickle=True)
    # for i in range(len(results)):
    #     print(len(results[i]))
    ep_shift = len(results[0]) - len(results[1]) + 1
    if len(results) > 4:
        fig, ax = plt.subplots(1,3, figsize=(18,6))
    else:
        fig, ax = plt.subplots(1,2, figsize=(15,6))
    ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
    ax[0].plot(range(ep_shift, len(results[1])+ep_shift), results[1], label='Validation error')
    ax[0].set_xticks(np.arange(1, len(results[0])+1))
    ax[0].set_xlabel('Training epochs', fontsize=16)
    ax[0].set_ylabel('Cross entropy', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[1].plot(range(1, len(results[2])+1), results[2], label='Training accuracy')
    ax[1].plot(range(ep_shift, len(results[3])+ep_shift), results[3], label='Validation accuracy')
    ax[1].set_xticks(np.arange(1, len(results[2])+1))
    # ax[1].plot(range(1, len(results[1])+1), results[1], label='Training accuracy')
    # ax[1].set_xticks(np.arange(1, len(results[1])+1))
    ax[1].set_xlabel('Training epochs', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].legend(fontsize=16)
    if len(results) > 4:
        ax[2].plot(range(1, len(results[4])+1), results[4], label='cos_sim_h1h2_list')
        ax[2].set_xticks(np.arange(1, len(results[4])+1))
        ax[2].set_xlabel('Training epochs', fontsize=16)
        ax[2].set_ylabel('cos_sim_h1h2_list', fontsize=16)
        ax[2].legend(fontsize=16)
    plt.suptitle('Learning Curve', fontsize=20)
    fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'learning_curve.jpg'),
                bbox_inches='tight')
    plt.close(fig)

# Train the hallucinator
def inference(args):
    print('==================== inference ====================')
    train_path = os.path.join(args.result_path, args.json_path, 'base.json')
    test_path = os.path.join(args.result_path, args.json_path, 'novel.json')
    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.GAN:
            # HAL_PN_GAN: implementation of (Y.-X. Wang, CVPR 2018)
            print('train_hallucinator.py --> main() --> inference(): use HAL_PN_GAN')
            net = HAL_PN_GAN(sess,
                             model_name=args.hallucinator_name,
                             result_path=args.result_path,
                             train_path=train_path,
                             test_path=test_path,
                             label_key=args.label_key,
                             bsize=args.bsize,
                             n_way=args.n_way,
                             n_shot=args.n_shot,
                             n_aug=args.n_aug,
                             n_query_all=args.n_query_all,
                             n_way_t=args.n_way_t,
                             n_shot_t=args.n_shot_t,
                             n_aug_t=args.n_aug_t,
                             n_query_all_t=args.n_query_all_t,
                             fc_dim=args.fc_dim,
                             img_size=args.img_size,
                             c_dim=args.c_dim,
                             l2scale=args.l2scale,
                             n_train_class=args.n_base_class,
                             n_test_class=args.n_novel_class,
                             with_BN=args.with_BN,
                             with_pro=args.with_pro,
                             num_parallel_calls=args.num_parallel_calls)
        elif args.GAN2:
            # HAL_PN_GAN2: adds one more layer to the hallucinator of HAL_PN_GAN
            print('train_hallucinator.py --> main() --> inference(): use HAL_PN_GAN2')
            net = HAL_PN_GAN2(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              test_path=test_path,
                              label_key=args.label_key,
                              bsize=args.bsize,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              n_way_t=args.n_way_t,
                              n_shot_t=args.n_shot_t,
                              n_aug_t=args.n_aug_t,
                              n_query_all_t=args.n_query_all_t,
                              fc_dim=args.fc_dim,
                              img_size=args.img_size,
                              c_dim=args.c_dim,
                              l2scale=args.l2scale,
                              n_train_class=args.n_base_class,
                              n_test_class=args.n_novel_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls)
        elif args.AFHN:
            print('train_hallucinator.py --> main() --> inference(): use HAL_PN_AFHN')
            net = HAL_PN_AFHN(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              test_path=test_path,
                              label_key=args.label_key,
                              bsize=args.bsize,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              n_way_t=args.n_way_t,
                              n_shot_t=args.n_shot_t,
                              n_aug_t=args.n_aug_t,
                              n_query_all_t=args.n_query_all_t,
                              fc_dim=args.fc_dim,
                              img_size=args.img_size,
                              c_dim=args.c_dim,
                              lambda_meta=args.lambda_meta,
                              lambda_tf=args.lambda_tf,
                              lambda_ar=args.lambda_ar,
                              l2scale=args.l2scale,
                              n_train_class=args.n_base_class,
                              n_test_class=args.n_novel_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls)
        elif args.PoseRef:
            print('train_hallucinator.py --> main() --> inference(): use HAL_PN_PoseRef')
            net = HAL_PN_PoseRef(sess,
                                 model_name=args.hallucinator_name,
                                 result_path=args.result_path,
                                 train_path=train_path,
                                 test_path=test_path,
                                 label_key=args.label_key,
                                 bsize=args.bsize,
                                 n_way=args.n_way,
                                 n_shot=args.n_shot,
                                 n_aug=args.n_aug,
                                 n_query_all=args.n_query_all,
                                 n_way_t=args.n_way_t,
                                 n_shot_t=args.n_shot_t,
                                 n_aug_t=args.n_aug_t,
                                 n_query_all_t=args.n_query_all_t,
                                 fc_dim=args.fc_dim,
                                 img_size=args.img_size,
                                 c_dim=args.c_dim,
                                 lambda_meta=args.lambda_meta,
                                 lambda_recon=args.lambda_recon,
                                 lambda_consistency=args.lambda_consistency,
                                 lambda_consistency_pose=args.lambda_consistency_pose,
                                 lambda_intra=args.lambda_intra,
                                 lambda_pose_code_reg=args.lambda_pose_code_reg,
                                 lambda_aux=args.lambda_aux,
                                 lambda_gan=args.lambda_gan,
                                 lambda_tf=args.lambda_tf,
                                 l2scale=args.l2scale,
                                 n_train_class=args.n_base_class,
                                 n_test_class=args.n_novel_class,
                                 with_BN=args.with_BN,
                                 with_pro=args.with_pro,
                                 num_parallel_calls=args.num_parallel_calls)
        else:
            print('train_hallucinator.py --> main() --> inference(): use HAL_PN_baseline')
            net = HAL_PN_baseline(sess,
                                  model_name=args.hallucinator_name,
                                  result_path=args.result_path,
                                  train_path=train_path,
                                  test_path=test_path,
                                  label_key=args.label_key,
                                  bsize=args.bsize,
                                  n_way=args.n_way,
                                  n_shot=args.n_shot,
                                  n_query_all=args.n_query_all,
                                  n_way_t=args.n_way_t,
                                  n_shot_t=args.n_shot_t,
                                  n_query_all_t=args.n_query_all_t,
                                  fc_dim=args.fc_dim,
                                  img_size=args.img_size,
                                  c_dim=args.c_dim,
                                  l2scale=args.l2scale,
                                  n_train_class=args.n_base_class,
                                  n_test_class=args.n_novel_class,
                                  with_BN=args.with_BN,
                                  with_pro=args.with_pro,
                                  num_parallel_calls=args.num_parallel_calls)
        all_vars, trainable_vars, all_regs = net.build_model()
        res = net.inference(hal_from=os.path.join(args.result_path, args.hallucinator_name, 'models'),
                            label_key=args.label_key,
                            n_ite_per_epoch=args.n_ite_per_epoch)

if __name__ == '__main__':
    main()
