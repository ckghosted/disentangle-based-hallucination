import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_hal import HAL_PN_only, HAL_PN_GAN, HAL_PN_GAN2, HAL_PN_AFHN, HAL_PN_PoseRef, HAL_PN_PoseRef_Before
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
    parser.add_argument('--extractor_folder', type=str, help='Folder name of the saved feature extracted by the pre-trained backbone')
    parser.add_argument('--hallucinator_name', type=str, help='Folder name to save hallucinator models and learning curves')
    parser.add_argument('--image_path', default='./image_folder', type=str, help='Path to the images for visualization')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--n_way', default=40, type=int, help='Number of classes in the support set')
    parser.add_argument('--n_shot', default=5, type=int, help='Number of samples per class in the support set')
    parser.add_argument('--n_aug', default=20, type=int, help='Number of samples per class in the augmented support set')
    parser.add_argument('--n_query_all', default=100, type=int, help='Number of samples in the query set')
    parser.add_argument('--n_intra', default=0, type=int, help='Number of intra-class samples for each class in the support set')
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr_start', default=1e-5, type=float, help='Initial learning rate for episodic training')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='Learning rate decay factor for episodic training')
    parser.add_argument('--lr_decay_step', default=0, type=int, help='Number of epochs per learning rate decay for episodic training, default 0: use num_epoch//3')
    parser.add_argument('--patience', default=20, type=int, help='Patience for early-stop mechanism')
    parser.add_argument('--n_ite_per_epoch', default=600, type=int, help='Number of iterations (episodes) per epoch')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--z_dim', default=512, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
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
    parser.add_argument('--n_train_class', default=64, type=int, help='Number of base class')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--AFHN', action='store_true', help='Use AFHN if present')
    parser.add_argument('--PoseRef', action='store_true', help='Use pose-ref-based hallucinator if present')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    parser.add_argument('--label_key', default='image_labels', type=str, help='image_labels or image_labels_id')
    parser.add_argument('--gpu_frac', default=0.5, type=float, help='per_process_gpu_memory_fraction (0.0~1.0)')
    parser.add_argument('--with_BN', action='store_true', help='Use batch_norm() in the feature extractor mode if present')
    parser.add_argument('--with_pro', action='store_true', help='Use additional embedding network for prototypical network if present')
    parser.add_argument('--num_parallel_calls', default=4, type=int, help='Number of core used to prepare data')
    parser.add_argument('--exp_tag', type=str, help='cv, final, or common')
    parser.add_argument('--ave_before_encode', action='store_true', help='Use class HAL_PN_PoseRef_Before (take feature average before class encoder) if present')
    parser.add_argument('--run_validation', action='store_true', help='Use val_train_feat to pick the best hallucinator if present')
    parser.add_argument('--d_per_g', default=5, type=int, help='Number of discriminator updates per generator update')
    parser.add_argument('--n_gallery_per_class', default=0, type=int, help='Number of gallery samples per class (default 0: use the whole base-class dataset as both gallery and probe)')
    args = parser.parse_args()
    train(args)
    # extract_pose(args)

# Train the hallucinator
def train(args):
    print('==================== training ====================')
    if args.exp_tag == 'cv':
        train_pickled_fname = 'cv_base_train_feat'
    elif args.exp_tag == 'final':
        train_pickled_fname = 'final_base_train_feat'
    elif args.exp_tag == 'common':
        train_pickled_fname = 'base_train_feat'
        val_pickled_fname = 'val_train_feat'
    train_path = os.path.join(args.result_path, args.extractor_folder, train_pickled_fname)
    if args.run_validation:
        val_path = os.path.join(args.result_path, args.extractor_folder, val_pickled_fname)
    else:
        val_path = None
    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.GAN:
            # HAL_PN_GAN: implementation of (Y.-X. Wang, CVPR 2018)
            print('train_hal.py --> main() --> train(): use HAL_PN_GAN')
            net = HAL_PN_GAN(sess,
                             model_name=args.hallucinator_name,
                             result_path=args.result_path,
                             train_path=train_path,
                             val_path=val_path,
                             label_key=args.label_key,
                             n_way=args.n_way,
                             n_shot=args.n_shot,
                             n_aug=args.n_aug,
                             n_query_all=args.n_query_all,
                             fc_dim=args.fc_dim,
                             z_dim=args.z_dim,
                             z_std=args.z_std,
                             l2scale=args.l2scale,
                             n_train_class=args.n_train_class,
                             with_BN=args.with_BN,
                             with_pro=args.with_pro,
                             num_parallel_calls=args.num_parallel_calls,
                             lambda_meta=args.lambda_meta,
                             lambda_tf=args.lambda_tf,
                             lambda_ar=args.lambda_ar)
        elif args.GAN2:
            # HAL_PN_GAN2: adds one more layer to the hallucinator of HAL_PN_GAN
            print('train_hal.py --> main() --> train(): use HAL_PN_GAN2')
            net = HAL_PN_GAN2(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              val_path=val_path,
                              label_key=args.label_key,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              fc_dim=args.fc_dim,
                              z_dim=args.z_dim,
                              z_std=args.z_std,
                              l2scale=args.l2scale,
                              n_train_class=args.n_train_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls,
                              lambda_meta=args.lambda_meta,
                              lambda_tf=args.lambda_tf,
                              lambda_ar=args.lambda_ar)
        elif args.AFHN:
            print('train_hal.py --> main() --> train(): use HAL_PN_AFHN')
            net = HAL_PN_AFHN(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              val_path=val_path,
                              label_key=args.label_key,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_aug=args.n_aug,
                              n_query_all=args.n_query_all,
                              fc_dim=args.fc_dim,
                              z_dim=args.z_dim,
                              z_std=args.z_std,
                              l2scale=args.l2scale,
                              n_train_class=args.n_train_class,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro,
                              num_parallel_calls=args.num_parallel_calls,
                              lambda_meta=args.lambda_meta,
                              lambda_tf=args.lambda_tf,
                              lambda_ar=args.lambda_ar)
        elif args.PoseRef:
            if args.ave_before_encode:
                print('train_hal.py --> main() --> train(): use HAL_PN_PoseRef_Before')
                net = HAL_PN_PoseRef_Before(sess,
                                     model_name=args.hallucinator_name,
                                     result_path=args.result_path,
                                     train_path=train_path,
                                     val_path=val_path,
                                     label_key=args.label_key,
                                     n_way=args.n_way,
                                     n_shot=args.n_shot,
                                     n_aug=args.n_aug,
                                     n_query_all=args.n_query_all,
                                     n_intra=args.n_intra,
                                     fc_dim=args.fc_dim,
                                     l2scale=args.l2scale,
                                     n_train_class=args.n_train_class,
                                     with_BN=args.with_BN,
                                     with_pro=args.with_pro,
                                     num_parallel_calls=args.num_parallel_calls,
                                     lambda_meta=args.lambda_meta,
                                     lambda_recon=args.lambda_recon,
                                     lambda_consistency=args.lambda_consistency,
                                     lambda_consistency_pose=args.lambda_consistency_pose,
                                     lambda_intra=args.lambda_intra,
                                     lambda_pose_code_reg=args.lambda_pose_code_reg,
                                     lambda_aux=args.lambda_aux,
                                     lambda_gan=args.lambda_gan,
                                     lambda_tf=args.lambda_tf,
                                     d_per_g=args.d_per_g,
                                     n_gallery_per_class=args.n_gallery_per_class)
            else:
                print('train_hal.py --> main() --> train(): use HAL_PN_PoseRef')
                net = HAL_PN_PoseRef(sess,
                                     model_name=args.hallucinator_name,
                                     result_path=args.result_path,
                                     train_path=train_path,
                                     val_path=val_path,
                                     label_key=args.label_key,
                                     n_way=args.n_way,
                                     n_shot=args.n_shot,
                                     n_aug=args.n_aug,
                                     n_query_all=args.n_query_all,
                                     n_intra=args.n_intra,
                                     fc_dim=args.fc_dim,
                                     l2scale=args.l2scale,
                                     n_train_class=args.n_train_class,
                                     with_BN=args.with_BN,
                                     with_pro=args.with_pro,
                                     num_parallel_calls=args.num_parallel_calls,
                                     lambda_meta=args.lambda_meta,
                                     lambda_recon=args.lambda_recon,
                                     lambda_consistency=args.lambda_consistency,
                                     lambda_consistency_pose=args.lambda_consistency_pose,
                                     lambda_intra=args.lambda_intra,
                                     lambda_pose_code_reg=args.lambda_pose_code_reg,
                                     lambda_aux=args.lambda_aux,
                                     lambda_gan=args.lambda_gan,
                                     lambda_tf=args.lambda_tf,
                                     d_per_g=args.d_per_g,
                                     n_gallery_per_class=args.n_gallery_per_class)
        else:
            # print('train_hal.py --> main() --> train(): use HAL_PN_baseline')
            # print('No HAL_PN_baseline for few-shot multiclass classification experiments!')
            print('train_hal.py --> main() --> train(): use HAL_PN_only')
            net = HAL_PN_only(sess,
                              model_name=args.hallucinator_name,
                              result_path=args.result_path,
                              train_path=train_path,
                              val_path=val_path,
                              label_key=args.label_key,
                              n_way=args.n_way,
                              n_shot=args.n_shot,
                              n_query_all=args.n_query_all,
                              fc_dim=args.fc_dim,
                              l2scale=args.l2scale,
                              n_train_class=args.n_train_class,
                              with_BN=args.with_BN,
                              num_parallel_calls=args.num_parallel_calls,)
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
        if os.path.exists(args.image_path):
            image_path = args.image_path
        else:
            image_path = './image_folder'
        if args.lr_decay_step == 0:
            lr_decay_step = args.num_epoch//3
        else:
            lr_decay_step = args.lr_decay_step
        res = net.train(image_path=image_path,
                        num_epoch=args.num_epoch,
                        n_ite_per_epoch=args.n_ite_per_epoch,
                        lr_start=args.lr_start,
                        lr_decay=args.lr_decay,
                        lr_decay_step=lr_decay_step,
                        patience=args.patience)
    np.save(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'), res)
    
    # Plot learning curve
    results = np.load(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'))
    if args.run_validation:
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
        ax[0].plot(range(1, len(results[2])+1), results[2], label='Validation error')
        ax[0].set_xticks(np.arange(1, len(results[0])+1))
        ax[0].set_xlabel('Training epochs', fontsize=16)
        ax[0].set_ylabel('Cross entropy', fontsize=16)
        ax[0].legend(fontsize=16)
        ax[1].plot(range(1, len(results[1])+1), results[1], label='Training accuracy')
        ax[1].plot(range(1, len(results[3])+1), results[3], label='Validation accuracy')
        ax[1].set_xticks(np.arange(1, len(results[1])+1))
        ax[1].set_xlabel('Training epochs', fontsize=16)
        ax[1].set_ylabel('Accuracy', fontsize=16)
        ax[1].legend(fontsize=16)
        plt.suptitle('Learning Curve', fontsize=20)
        fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'learning_curve.jpg'),
                    bbox_inches='tight')
        plt.close(fig)
    else:
        fig, ax = plt.subplots(1,2, figsize=(15,6))
        ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
        ax[0].set_xticks(np.arange(1, len(results[0])+1))
        ax[0].set_xlabel('Training epochs', fontsize=16)
        ax[0].set_ylabel('Cross entropy', fontsize=16)
        ax[0].legend(fontsize=16)
        ax[1].plot(range(1, len(results[1])+1), results[1], label='Training accuracy')
        ax[1].set_xticks(np.arange(1, len(results[1])+1))
        ax[1].set_xlabel('Training epochs', fontsize=16)
        ax[1].set_ylabel('Accuracy', fontsize=16)
        ax[1].legend(fontsize=16)
        plt.suptitle('Learning Curve', fontsize=20)
        fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'learning_curve.jpg'),
                    bbox_inches='tight')
        plt.close(fig)
    if args.GAN or args.GAN2 or args.AFHN:
        cos_sim_h1h2_list = results[-1]
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.plot(range(1, len(cos_sim_h1h2_list)+1), cos_sim_h1h2_list, label='cos_sim_h1h2_list')
        ax.set_xticks(np.arange(1, len(cos_sim_h1h2_list)+1))
        ax.set_xlabel('Training epochs', fontsize=16)
        ax.set_ylabel('Cosine similarity', fontsize=16)
        ax.legend(fontsize=16)
        plt.suptitle('Cosine similarity', fontsize=20)
        fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'cos_sim_h1h2.jpg'),
                    bbox_inches='tight')
        plt.close(fig)

def extract_pose(args):
    print('==================== extract_pose ====================')
    if args.exp_tag == 'cv':
        train_pickled_fname = 'cv_base_train_feat'
    elif args.exp_tag == 'final':
        train_pickled_fname = 'final_base_train_feat'
    elif args.exp_tag == 'common':
        train_pickled_fname = 'base_train_feat'
        val_pickled_fname = 'val_train_feat'
    train_path = os.path.join(args.result_path, args.extractor_folder, train_pickled_fname)
    if args.run_validation:
        val_path = os.path.join(args.result_path, args.extractor_folder, val_pickled_fname)
    else:
        val_path = None
    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.PoseRef:
            if args.ave_before_encode:
                print('train_hal.py --> main() --> train(): use HAL_PN_PoseRef_Before')
                net = HAL_PN_PoseRef_Before(sess,
                                     model_name=args.hallucinator_name,
                                     result_path=args.result_path,
                                     train_path=train_path,
                                     val_path=val_path,
                                     label_key=args.label_key,
                                     n_way=args.n_way,
                                     n_shot=args.n_shot,
                                     n_aug=args.n_aug,
                                     n_query_all=args.n_query_all,
                                     n_intra=args.n_intra,
                                     fc_dim=args.fc_dim,
                                     l2scale=args.l2scale,
                                     n_train_class=args.n_train_class,
                                     with_BN=args.with_BN,
                                     with_pro=args.with_pro,
                                     num_parallel_calls=args.num_parallel_calls,
                                     lambda_meta=args.lambda_meta,
                                     lambda_recon=args.lambda_recon,
                                     lambda_consistency=args.lambda_consistency,
                                     lambda_consistency_pose=args.lambda_consistency_pose,
                                     lambda_intra=args.lambda_intra,
                                     lambda_pose_code_reg=args.lambda_pose_code_reg,
                                     lambda_aux=args.lambda_aux,
                                     lambda_gan=args.lambda_gan,
                                     lambda_tf=args.lambda_tf,
                                     d_per_g=args.d_per_g,
                                     n_gallery_per_class=args.n_gallery_per_class)
            else:
                print('train_hal.py --> main() --> train(): use HAL_PN_PoseRef')
                net = HAL_PN_PoseRef(sess,
                                     model_name=args.hallucinator_name,
                                     result_path=args.result_path,
                                     train_path=train_path,
                                     val_path=val_path,
                                     label_key=args.label_key,
                                     n_way=args.n_way,
                                     n_shot=args.n_shot,
                                     n_aug=args.n_aug,
                                     n_query_all=args.n_query_all,
                                     n_intra=args.n_intra,
                                     fc_dim=args.fc_dim,
                                     l2scale=args.l2scale,
                                     n_train_class=args.n_train_class,
                                     with_BN=args.with_BN,
                                     with_pro=args.with_pro,
                                     num_parallel_calls=args.num_parallel_calls,
                                     lambda_meta=args.lambda_meta,
                                     lambda_recon=args.lambda_recon,
                                     lambda_consistency=args.lambda_consistency,
                                     lambda_consistency_pose=args.lambda_consistency_pose,
                                     lambda_intra=args.lambda_intra,
                                     lambda_pose_code_reg=args.lambda_pose_code_reg,
                                     lambda_aux=args.lambda_aux,
                                     lambda_gan=args.lambda_gan,
                                     lambda_tf=args.lambda_tf,
                                     d_per_g=args.d_per_g,
                                     n_gallery_per_class=args.n_gallery_per_class)
            all_vars, trainable_vars, all_regs = net.build_model()
            hal_from = os.path.join(args.result_path, args.hallucinator_name, 'models_hal_pro')
            net.extract_pose(hal_from=hal_from,
                             bsize=1000)

if __name__ == '__main__':
    main()
