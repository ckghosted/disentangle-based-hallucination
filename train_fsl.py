import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from model_fsl import FSL, MSL, MSL_PN
from model_fsl import FSL_PN_GAN, FSL_PN_GAN2
from model_fsl import FSL_PN_AFHN
from model_fsl import FSL_PN_DFHN
import re, glob

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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.stats import entropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--model_name', type=str, help='Folder name to save FSL models and learning curves')
    parser.add_argument('--extractor_folder', type=str, help='Folder name of the saved feature extracted by the pre-trained backbone')
    parser.add_argument('--hallucinator_name', default='Baseline', type=str, help='Folder name of the saved hallucinator model (default Baseline: no hallucination)')
    parser.add_argument('--hal_epoch', default=0, type=int, help='Hallucinator version (number of epoch), default 0: use the latest version (None)')
    parser.add_argument('--image_path', default='./image_folder', type=str, help='Path to the images for visualization')
    parser.add_argument('--n_class', default=100, type=int, help='Number of all classes (base + validation or base + novel)')
    parser.add_argument('--n_base_class', default=100, type=int, help='Number of base classes')
    parser.add_argument('--n_shot', default=1, type=int, help='Number of shot')
    parser.add_argument('--n_aug', default=40, type=int, help='Number of samples per training class AFTER hallucination')
    parser.add_argument('--n_top', default=5, type=int, help='Number to compute the top-n accuracy')
    parser.add_argument('--num_ite', default=1000, type=int, help='Number of iterations for training the final linear classifier')
    parser.add_argument('--bsize', default=100, type=int, help='Batch size for training the final linear classifier')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate for training the final linear classifier')
    parser.add_argument('--l2scale', default=0.0, type=float, help='L2-regularizer scale for training the final linear classifier')
    parser.add_argument('--fc_dim', default=512, type=int, help='Feature dimension')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    parser.add_argument('--label_key', default='image_labels', type=str, help='image_labels or image_labels_id')
    parser.add_argument('--exp_tag', type=str, help='cv or final for imagenet-1k; val or novel for other datasets')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN-based hallucinator (with one more layer in the hallucinator) if present')
    parser.add_argument('--AFHN', action='store_true', help='Use AFHN if present')
    parser.add_argument('--DFHN', action='store_true', help='Use DFHN if present')
    parser.add_argument('--MSL', action='store_true', help='Train MSL (many-shot learning) classifier using features if present')
    parser.add_argument('--MSL_PN', action='store_true', help='Train MSL (many-shot learning) classifier using class codes if present')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
    parser.add_argument('--z_std', default=1.0, type=float, help='Standard deviation of the input noise to the GAN-based hallucinator')
    parser.add_argument('--gpu_frac', default=1.0, type=float, help='per_process_gpu_memory_fraction (0.0~1.0)')
    parser.add_argument('--with_BN', action='store_true', help='Use batch_norm() in the feature extractor mode if present')
    parser.add_argument('--with_pro', action='store_true', help='Use additional embedding network for prototypical network if present')
    parser.add_argument('--ite_idx', default=0, type=int, help='iteration index for imagenet-1k dataset (0, 1, 2, 3, 4)')
    parser.add_argument('--n_gallery_per_class', default=0, type=int, help='Number of samples per class in the gallery set, default 0: use the whole base-class dataset')
    parser.add_argument('--n_base_lb_per_novel', default=0, type=int, help='number of base classes as the candidates of pose-ref for each novel class')
    parser.add_argument('--use_canonical_gallery', action='store_true', help='Use gallery_indexes_canonical_XX.npy if present')
    parser.add_argument('--n_clusters_per_class', default=0, type=int, help='Number of clusters per base class for making the gallery set')
    parser.add_argument('--test_mode', action='store_true', help='After training, return novel class code and pose code if present')
    
    args = parser.parse_args()
    train(args)
    inference(args)
    if args.test_mode and args.ite_idx == 0:
        visualize(args)

# Hallucination and training
def train(args):
    print('============================ train ============================')
    novel_idx = None
    if args.exp_tag == 'cv':
        train_novel_fname = 'cv_novel_train_feat'
        train_base_fname = 'cv_base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'cv_novel_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'cv_novel_idx_%d.npy' % args.ite_idx))
    elif args.exp_tag == 'final':
        train_novel_fname = 'final_novel_train_feat'
        train_base_fname = 'final_base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'final_novel_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'final_novel_idx_%d.npy' % args.ite_idx))
    elif args.exp_tag == 'val':
        train_novel_fname = 'val_train_feat'
        train_base_fname = 'base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'selected_val_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'selected_val_idx_%d.npy' % args.ite_idx))
    elif args.exp_tag == 'novel':
        train_novel_fname = 'novel_train_feat'
        train_base_fname = 'base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'selected_novel_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'selected_novel_idx_%d.npy' % args.ite_idx))
    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.GAN:
            net = FSL_PN_GAN(sess,
                             model_name=args.model_name,
                             result_path=os.path.join(args.result_path, args.hallucinator_name),
                             fc_dim=args.fc_dim,
                             n_class=args.n_class,
                             n_base_class=args.n_base_class,
                             l2scale=args.l2scale,
                             z_dim=args.z_dim,
                             z_std=args.z_std)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std)
        elif args.AFHN:
            net = FSL_PN_AFHN(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std)
        elif args.DFHN:
            net = FSL_PN_DFHN(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              n_gallery_per_class=args.n_gallery_per_class,
                              n_base_lb_per_novel=args.n_base_lb_per_novel,
                              use_canonical_gallery=args.use_canonical_gallery,
                              n_clusters_per_class=args.n_clusters_per_class)
        elif args.MSL:
            net = MSL(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      n_class=args.n_class,
                      n_base_class=args.n_base_class,
                      l2scale=args.l2scale)
        elif args.MSL_PN:
            net = MSL_PN(sess,
                         model_name=args.model_name,
                         result_path=os.path.join(args.result_path, args.hallucinator_name),
                         fc_dim=args.fc_dim,
                         n_class=args.n_class,
                         n_base_class=args.n_base_class,
                         l2scale=args.l2scale,
                         with_BN=args.with_BN)
        else:
            net = FSL(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      n_class=args.n_class,
                      n_base_class=args.n_base_class,
                      l2scale=args.l2scale)
        all_vars, trainable_vars, all_regs = net.build_model()
        ### specify the hallucinator use to augment novel data
        hal_from = os.path.join(args.result_path, args.hallucinator_name, 'models_hal_pro')
        if args.hal_epoch > 0:
            hal_from_ckpt = args.hallucinator_name+'.model-hal-pro-%d' % args.hal_epoch
            # print()
            # print('===========================================================================================')
            # print('hal_from:', hal_from)
            # print(os.path.exists(hal_from))
            # print('hal_from_ckpt:', hal_from_ckpt)
            # print(os.path.exists(hal_from_ckpt))
            # print('===========================================================================================')
            # print()
        else:
            hal_from_ckpt = None
        res = net.train(train_novel_path=os.path.join(args.result_path, args.extractor_folder, train_novel_fname),
                        train_base_path=os.path.join(args.result_path, args.extractor_folder, train_base_fname),
                        hal_from=hal_from,
                        hal_from_ckpt=hal_from_ckpt,
                        image_path=args.image_path,
                        label_key=args.label_key,
                        n_shot=args.n_shot,
                        n_aug=args.n_aug,
                        n_top=args.n_top,
                        bsize=args.bsize,
                        learning_rate=args.learning_rate,
                        num_ite=args.num_ite,
                        novel_idx=novel_idx,
                        test_mode=args.test_mode)
    ## debug mode
    if args.debug or args.test_mode:
        ### (1) check trainable variables and regularizers
        print('------------------[all_vars]------------------')
        for var in all_vars:
            print(var.name)
        print('------------------[trainable_vars]------------------')
        for var in trainable_vars:
            print(var.name)
        print('------------------[all_regs]------------------')
        for var in all_regs:
            print(var.name)
        ### (2) save statistics and hallucinated features, and plot learning curve
        np.save(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'), res)
        results = np.load(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'), allow_pickle=True)
        fig, ax = plt.subplots(1, 2, figsize=(15,6))
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
        fig.savefig(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'learning_curve.jpg'),
                    bbox_inches='tight')
        plt.close(fig)

# Inference
def inference(args):
    print('============================ inference ============================')
    if args.exp_tag == 'cv':
        test_novel_fname = 'cv_novel_test_feat'
        test_base_fname = 'cv_base_test_feat'
    elif args.exp_tag == 'final':
        test_novel_fname = 'final_novel_test_feat'
        test_base_fname = 'final_base_test_feat'
    elif args.exp_tag == 'val':
        test_novel_fname = 'val_test_feat'
        test_base_fname = 'base_test_feat'
    elif args.exp_tag == 'novel':
        test_novel_fname = 'novel_test_feat'
        test_base_fname = 'base_test_feat'
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.GAN:
            net = FSL_PN_GAN(sess,
                             model_name=args.model_name,
                             result_path=os.path.join(args.result_path, args.hallucinator_name),
                             fc_dim=args.fc_dim,
                             n_class=args.n_class,
                             n_base_class=args.n_base_class,
                             l2scale=args.l2scale,
                             z_dim=args.z_dim,
                             z_std=args.z_std)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std)
        elif args.AFHN:
            net = FSL_PN_AFHN(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std)
        elif args.DFHN:
            net = FSL_PN_DFHN(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              n_gallery_per_class=args.n_gallery_per_class,
                              n_base_lb_per_novel=args.n_base_lb_per_novel,
                              use_canonical_gallery=args.use_canonical_gallery,
                              n_clusters_per_class=args.n_clusters_per_class)
        elif args.MSL:
            net = MSL(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      n_class=args.n_class,
                      n_base_class=args.n_base_class,
                      l2scale=args.l2scale)
        elif args.MSL_PN:
            net = MSL_PN(sess,
                         model_name=args.model_name,
                         result_path=os.path.join(args.result_path, args.hallucinator_name),
                         fc_dim=args.fc_dim,
                         n_class=args.n_class,
                         n_base_class=args.n_base_class,
                         l2scale=args.l2scale,
                         with_BN=args.with_BN)
        else:
            net = FSL(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      n_class=args.n_class,
                      n_base_class=args.n_base_class,
                      l2scale=args.l2scale)
        net.build_model()
        net.inference(test_novel_path=os.path.join(args.result_path, args.extractor_folder, test_novel_fname),
                      test_base_path=os.path.join(args.result_path, args.extractor_folder, test_base_fname),
                      gen_from=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'models'),
                      label_key=args.label_key,
                      out_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name),
                      n_top=args.n_top,
                      bsize=args.bsize)

# Visualization
def plot(samples, n_row, n_col, x_dim=84, title=None, subtitles=None, fontsize=20):
    fig = plt.figure(figsize=(n_col*2, n_row*2.2))
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.05, hspace=0.2)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if title is not None and i == 0:
            plt.title(title, fontsize=fontsize)
        elif subtitles is not None:
            plt.title(subtitles[i], fontsize=fontsize)
        plt.imshow(sample.reshape(x_dim, x_dim, 3))
    return fig

def dim_reduction(_input,
                  method='TSNE',
                  n_components=2,
                  perplexity=30,
                  n_iter=1000):
    if method == 'TSNE':
        return TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter).fit_transform(_input)
    elif method == 'PCA':
        return PCA(n_components=n_components).fit_transform(_input)

def plot_emb_results(_emb, # 2-dim feature 
                     _labels, # 1-dim label list of _emb (e.g., absolute novel labeling)
                     considered_lb,
                     all_labels,
                     n_shot,
                     n_aug,
                     color_list=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
                     plot_seed=True,
                     plot_real=True,
                     plot_hal=True,
                     plot_others_in_white=False, # plot all other labels (not in considered_lb) in white (won't show) to keep layout unchanged as if all labels are plotted
                     fig_dim=12,
                     title='Hallucinated and real features',
                     save_path=None):
    fig = plt.figure(figsize=(fig_dim, fig_dim))
    color_idx = 0
    plt_lb = []
    for lb in all_labels:
        idx_for_this_lb = [i for i in range(_labels.shape[0]) if _labels[i] == lb]
        X_embedded_lb = _emb[idx_for_this_lb,:]
        n_feat_per_lb = len(idx_for_this_lb)
        if lb in considered_lb:
            #### plot seed features as stars
            if plot_seed:
                plt.scatter(x=X_embedded_lb[0:n_shot, 0],
                            y=X_embedded_lb[0:n_shot, 1],
                            marker='*',
                            color=color_list[color_idx],
                            alpha=0.7,
                            s=200)
            #### plot all real features as crosses
            if plot_real:
                plt.scatter(x=X_embedded_lb[n_aug:n_feat_per_lb, 0],
                            y=X_embedded_lb[n_aug:n_feat_per_lb, 1],
                            color=color_list[color_idx],
                            alpha=0.4,
                            marker='x',
                            s=30)
            #### plot hallucinated features as triangles
            if plot_hal:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_aug, 0],
                                          y=X_embedded_lb[n_shot:n_aug, 1],
                                          alpha=0.2,
                                          marker='^',
                                          color=color_list[color_idx],
                                          s=60))
            else:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_aug, 0],
                                          y=X_embedded_lb[n_shot:n_aug, 1],
                                          alpha=0.2,
                                          marker='^',
                                          color='white',
                                          s=60))
            color_idx += 1
        else:
            if plot_others_in_white:
                if plot_seed:
                    plt.scatter(x=X_embedded_lb[0:n_shot, 0],
                                y=X_embedded_lb[0:n_shot, 1],
                                marker='*',
                                color='white',
                                s=200)
                if plot_real:
                    plt.scatter(x=X_embedded_lb[n_aug:n_feat_per_lb, 0],
                                y=X_embedded_lb[n_aug:n_feat_per_lb, 1],
                                color='white',
                                marker='x',
                                s=30)
                plt.scatter(x=X_embedded_lb[n_shot:n_aug, 0],
                            y=X_embedded_lb[n_shot:n_aug, 1],
                            marker='^',
                            color='white',
                            s=30)
    plt.title(title, fontsize=20)
    plt.legend(plt_lb,
#                ['label %d (%s)' % (lb, meta_dict[b'fine_label_names'][lb].decode('ascii')) for lb in np.sort(considered_lb)],
               ['label %d' % lb for lb in all_labels if lb in considered_lb],
               fontsize=16, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.mean(np.square(array - value), axis=1)).argmin()
    diff = np.mean(np.square(array[idx] - value))
    return idx, diff

def visualize(args):
    print('============================ visualize ============================')
    if args.exp_tag == 'cv':
        train_novel_fname = 'cv_novel_train_feat'
        train_base_fname = 'cv_base_train_feat'
    elif args.exp_tag == 'final':
        train_novel_fname = 'final_novel_train_feat'
        train_base_fname = 'final_base_train_feat'
    elif args.exp_tag == 'val':
        train_novel_fname = 'val_train_feat'
        train_base_fname = 'base_train_feat'
    elif args.exp_tag == 'novel':
        train_novel_fname = 'novel_train_feat'
        train_base_fname = 'base_train_feat'
    ## (1) Load real features
    train_novel_dict = unpickle(os.path.join(args.result_path, args.extractor_folder, train_novel_fname))
    features_novel_train = train_novel_dict['features']
    labels_novel_train = train_novel_dict[args.label_key]
    fnames_novel_train = train_novel_dict['image_names']
    all_novel_labels = sorted(set(labels_novel_train))
    train_base_dict = unpickle(os.path.join(args.result_path, args.extractor_folder, train_base_fname))
    features_base_train = train_base_dict['features']
    labels_base_train = train_base_dict[args.label_key]
    fnames_base_train = train_base_dict['image_names']
    features_all_train = np.concatenate((features_novel_train, features_base_train), axis=0)
    labels_all_train = labels_novel_train + labels_base_train
    fnames_all_train = fnames_novel_train + fnames_base_train
    ## [2020/12/09] Frank suggests: search the nearest real feature for each hallucinated feature in the feature space FROM THE CORRECT LABEL!
    features_novel_train_each_lb = {}
    idxs_each_lb = {}
    for lb in all_novel_labels:
        temp_idxs = [i for i in range(features_novel_train.shape[0]) if labels_novel_train[i] == lb]
        idxs_each_lb[lb] = temp_idxs
        features_novel_train_each_lb[lb] = features_novel_train[temp_idxs,:]
    ## (2) Load training results
    training_results = np.load(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'), allow_pickle=True)
    final_novel_feat_dict = training_results[2]
    final_novel_class_code_dict = training_results[3]
    novel_code_class_all = training_results[4]
    base_code_class_all = training_results[5]
    code_class_all = np.concatenate((novel_code_class_all, base_code_class_all), axis=0)
    if len(training_results) > 6:
        final_novel_pose_code_dict = training_results[6]
        novel_code_pose_all = training_results[7]
        base_code_pose_all = training_results[8]
        code_pose_all = np.concatenate((novel_code_pose_all, base_code_pose_all), axis=0)
        pose_feat_dict = training_results[9]
    ## (3) Combine seed/hal features and real features
    # allow dimension reduction using only a subset of classes
    # lb_for_dim_reduction = all_novel_labels
    # lb_for_dim_reduction = [166, 50, 100, 152, 176]
    lb_for_dim_reduction = np.random.choice(all_novel_labels, 5, replace=False)
    X_all = None
    Y_all = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        feat_for_this_lb = np.concatenate((final_novel_feat_dict[lb], features_novel_train[idx_for_this_lb,:]), axis=0)
        labels_for_this_lb = np.repeat(lb, repeats=args.n_aug+len(idx_for_this_lb))
        if X_all is None:
            X_all = feat_for_this_lb
            Y_all = labels_for_this_lb
        else:
            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
    ## (4) Dimension reduction and visualization
    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
    considered_lb = lb_for_dim_reduction
    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                     _labels=Y_all,
                     considered_lb=considered_lb,
                     all_labels=all_novel_labels,
                     n_shot=args.n_shot,
                     n_aug=args.n_aug,
                     title='real and hallucinated features',
                     save_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'hal_vis.png'))
    ## (5) How about the prototypical-network-embeded features?
    X_all = None
    Y_all = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        feat_for_this_lb = np.concatenate((final_novel_class_code_dict[lb], novel_code_class_all[idx_for_this_lb,:]), axis=0)
        labels_for_this_lb = np.repeat(lb, repeats=args.n_aug+len(idx_for_this_lb))
        if X_all is None:
            X_all = feat_for_this_lb
            Y_all = labels_for_this_lb
        else:
            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
    considered_lb = lb_for_dim_reduction
    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                     _labels=Y_all,
                     considered_lb=considered_lb,
                     all_labels=all_novel_labels,
                     n_shot=args.n_shot,
                     n_aug=args.n_aug,
                     title='class codes of real and hallucinated features',
                     save_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'hal_vis_embedded.png'))
    ## (6) Find the closest real feature (and the corresponding image) for each hallucinated feature
    if len(training_results) > 6:
        corresponding_lb_DFHN_dict = {}
        for lb_idx in range(5):
            considered_lb = lb_for_dim_reduction[lb_idx]
            considered_feat_seed = final_novel_feat_dict[considered_lb][:args.n_shot,:]
            nearest_idx_seed, nearest_diff_seed = find_nearest_idx(features_novel_train, considered_feat_seed[0])
            considered_feat_pose = pose_feat_dict[considered_lb]
            nearest_indexes_pose = []
            nearest_differences_pose = []
            # (a) nearest in the feature space
            considered_feat_hal = final_novel_feat_dict[considered_lb][args.n_shot:,:]
            nearest_indexes_hal = []
            nearest_differences_hal = []
            # (b) nearest based on the class codes
            considered_feat_hal_class = final_novel_class_code_dict[considered_lb][args.n_shot:,:]
            nearest_indexes_hal_class = []
            nearest_differences_hal_class = []
            # (c) nearest based on the appearance codes
            considered_feat_hal_pose = final_novel_pose_code_dict[considered_lb][args.n_shot:,:]
            nearest_indexes_hal_pose = []
            nearest_differences_hal_pose = []
            n_hal_plot = min(8, args.n_aug-args.n_shot)
            for idx in range(n_hal_plot):
                nearest_idx, nearest_diff = find_nearest_idx(features_base_train, considered_feat_pose[idx])
                nearest_indexes_pose.append(nearest_idx)
                nearest_differences_pose.append(nearest_diff)
                # ##### (a) nearest in the feature space
                # ##### (option1) search from novel classes only
                # nearest_idx, nearest_diff = find_nearest_idx(features_novel_train, considered_feat_hal[idx])
                # nearest_indexes_hal.append(nearest_idx)
                # nearest_differences_hal.append(nearest_diff)
                # ##### (option2) search from all classes
                # nearest_idx, nearest_diff = find_nearest_idx(features_all_train, considered_feat_hal[idx])
                # nearest_indexes_hal.append(nearest_idx)
                # nearest_differences_hal.append(nearest_diff)
                # ##### (b) nearest based on the class codes
                # ##### (option1) search from novel classes only
                # nearest_idx, nearest_diff = find_nearest_idx(novel_code_class_all, considered_feat_hal_class[idx])
                # nearest_indexes_hal_class.append(nearest_idx)
                # nearest_differences_hal_class.append(nearest_diff)
                # ##### (option2) search from all classes
                # nearest_idx, nearest_diff = find_nearest_idx(code_class_all, considered_feat_hal_class[idx])
                # nearest_indexes_hal_class.append(nearest_idx)
                # nearest_differences_hal_class.append(nearest_diff)
                # ##### (c) nearest based on the appearance codes
                # ##### (option1) search from novel classes only
                # nearest_idx, nearest_diff = find_nearest_idx(novel_code_pose_all, considered_feat_hal_pose[idx])
                # nearest_indexes_hal_pose.append(nearest_idx)
                # nearest_differences_hal_pose.append(nearest_diff)
                # ##### (option2) search from all classes
                # nearest_idx, nearest_diff = find_nearest_idx(code_pose_all, considered_feat_hal_pose[idx])
                # nearest_indexes_hal_pose.append(nearest_idx)
                # nearest_differences_hal_pose.append(nearest_diff)
                ##### [2020/12/09] Frank suggests: search the nearest real feature for each hallucinated feature in the feature space FROM THE CORRECT LABEL!
                nearest_idx_, nearest_diff = find_nearest_idx(features_novel_train_each_lb[considered_lb], considered_feat_hal[idx])
                nearest_idx = idxs_each_lb[considered_lb][nearest_idx_]
                nearest_indexes_hal.append(nearest_idx)
                nearest_differences_hal.append(nearest_diff)

            x_dim = 84
            img_array = np.empty((3*n_hal_plot, x_dim, x_dim, 3), dtype='uint8')
            corresponding_lb_DFHN = []
            corresponding_lb_hal = []
            corresponding_lb_hal_class = []
            corresponding_lb_hal_pose = []
            for i in range(len(nearest_indexes_hal)):
                ##### (1) put seed image in the 1st row
                file_path = os.path.join(args.image_path, fnames_novel_train[nearest_idx_seed])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i,:] = img
                ##### -------------------------------------------------------
                ##### (2) put pose-ref image in the 2nd row
                idx = nearest_indexes_pose[i]
                file_path = os.path.join(args.image_path, fnames_base_train[idx])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i+len(nearest_indexes_hal),:] = img
                corresponding_lb_DFHN.append(str(labels_base_train[idx]))
                ##### -------------------------------------------------------
                ##### (3) put hal image (nearest searching from all classes in the feature space) in the 3rd row
                idx = nearest_indexes_hal[i]
                ##### (option1) search from novel classes only
                file_path = os.path.join(args.image_path, fnames_novel_train[idx])
                ##### (option2) search from all classes
                # file_path = os.path.join(args.image_path, fnames_all_train[idx])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i+2*len(nearest_indexes_hal),:] = img
                ##### (option1) search from novel classes only
                corresponding_lb_hal.append(str(labels_novel_train[idx]))
                ##### (option2) search from all classes
                # corresponding_lb_hal.append(str(labels_all_train[idx]))
                ##### -------------------------------------------------------
                ##### (4) put hal image (nearest searching from all classes based on the class codes) in the 4th row
                # idx = nearest_indexes_hal_class[i]
                ##### (option1) search from novel classes only
                # file_path = os.path.join(args.image_path, fnames_novel_train[idx])
                ##### (option2) search from all classes
                # file_path = os.path.join(args.image_path, fnames_all_train[idx])
                # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                # img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img_array[i+3*len(nearest_indexes_hal_class),:] = img
                ##### (option1) search from novel classes only
                # corresponding_lb_hal_class.append(str(labels_novel_train[idx]))
                ##### (option2) search from all classes
                # corresponding_lb_hal_class.append(str(labels_all_train[idx]))
                ##### -------------------------------------------------------
                ##### (5) put hal image (nearest searching from all classes based on the pose codes) in the 5th row
                # idx = nearest_indexes_hal_pose[i]
                ##### (option1) search from novel classes only
                # file_path = os.path.join(args.image_path, fnames_novel_train[idx])
                ##### (option2) search from all classes
                # file_path = os.path.join(args.image_path, fnames_all_train[idx])
                # img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                # img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img_array[i+4*len(nearest_indexes_hal_pose),:] = img
                ##### (option1) search from novel classes only
                # corresponding_lb_hal_pose.append(str(labels_novel_train[idx]))
                ##### (option2) search from all classes
                # corresponding_lb_hal_pose.append(str(labels_all_train[idx]))
            subtitle_list = [str(labels_novel_train[nearest_idx_seed]) for _ in range(n_hal_plot)] + corresponding_lb_DFHN + corresponding_lb_hal
            fig = plot(img_array, 3, n_hal_plot, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
            plt.savefig(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'nearest_images_%3d.png' % considered_lb), bbox_inches='tight')
            corresponding_lb_DFHN_dict[considered_lb] = corresponding_lb_DFHN
    else:
        for lb_idx in range(5):
            considered_lb = lb_for_dim_reduction[lb_idx]
            considered_feat_seed = final_novel_feat_dict[considered_lb][:args.n_shot,:]
            nearest_idx_seed, nearest_diff_seed = find_nearest_idx(features_novel_train, considered_feat_seed[0])

            # (a) nearest in the feature space
            considered_feat_hal = final_novel_feat_dict[considered_lb][args.n_shot:,:]
            nearest_indexes_hal = []
            nearest_differences_hal = []
            # (b) nearest based on the class codes
            considered_feat_hal_class = final_novel_class_code_dict[considered_lb][args.n_shot:,:]
            nearest_indexes_hal_class = []
            nearest_differences_hal_class = []
            n_hal_plot = min(8, args.n_aug-args.n_shot)
            for idx in range(n_hal_plot):
                # (a) nearest in the feature space
                # nearest_idx, nearest_diff = find_nearest_idx(features_novel_train, considered_feat_hal[idx])
                # nearest_indexes_hal.append(nearest_idx)
                # nearest_differences_hal.append(nearest_diff)
                nearest_idx, nearest_diff = find_nearest_idx(features_all_train, considered_feat_hal[idx])
                nearest_indexes_hal.append(nearest_idx)
                nearest_differences_hal.append(nearest_diff)
                # (b) nearest based on the class codes
                # nearest_idx, nearest_diff = find_nearest_idx(novel_code_class_all, considered_feat_hal_class[idx])
                # nearest_indexes_hal_class.append(nearest_idx)
                # nearest_differences_hal_class.append(nearest_diff)
                nearest_idx, nearest_diff = find_nearest_idx(code_class_all, considered_feat_hal_class[idx])
                nearest_indexes_hal_class.append(nearest_idx)
                nearest_differences_hal_class.append(nearest_diff)
            
            x_dim = 84
            img_array = np.empty((3*n_hal_plot, x_dim, x_dim, 3), dtype='uint8')
            corresponding_lb_hal = []
            corresponding_lb_hal_class = []
            for i in range(len(nearest_indexes_hal)):
                ## (1) put seed image in the 1st row
                file_path = os.path.join(args.image_path, fnames_novel_train[nearest_idx_seed])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i,:] = img
                ## (2) put hal image (nearest searching from all classes in the feature space) in the 2nd row
                idx = nearest_indexes_hal[i]
                # file_path = os.path.join(args.image_path, fnames_novel_train[idx])
                file_path = os.path.join(args.image_path, fnames_all_train[idx])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i+len(nearest_indexes_hal),:] = img
                # corresponding_lb_hal.append(str(labels_novel_train[idx]))
                corresponding_lb_hal.append(str(labels_all_train[idx]))
                ## (3) put hal image (nearest searching from all classes based on the class codes) in the 3rd row
                idx = nearest_indexes_hal_class[i]
                # file_path = os.path.join(args.image_path, fnames_novel_train[idx])
                file_path = os.path.join(args.image_path, fnames_all_train[idx])
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_array[i+2*len(nearest_indexes_hal_class),:] = img
                # corresponding_lb_hal_class.append(str(labels_novel_train[idx]))
                corresponding_lb_hal_class.append(str(labels_all_train[idx]))
            subtitle_list = [str(labels_novel_train[nearest_idx_seed]) for _ in range(n_hal_plot)] + corresponding_lb_hal + corresponding_lb_hal_class
            fig = plot(img_array, 3, n_hal_plot, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
            plt.savefig(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'nearest_images_%3d.png' % considered_lb), bbox_inches='tight')
    ## (7) Visualize class codes and pose codes
    X_code_class = None
    Y_code = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        class_code_for_this_lb = novel_code_class_all[idx_for_this_lb,:]
        labels_for_this_lb = np.repeat(lb, repeats=len(idx_for_this_lb))
        if X_code_class is None:
            X_code_class = class_code_for_this_lb
            Y_code = labels_for_this_lb
        else:
            X_code_class = np.concatenate((X_code_class, class_code_for_this_lb), axis=0)
            Y_code = np.concatenate((Y_code, labels_for_this_lb), axis=0)
    X_code_class_emb_TSNE30_1000 = dim_reduction(X_code_class, 'TSNE', 2, 30, 1000)
    considered_lb = lb_for_dim_reduction
    plot_emb_results(_emb=X_code_class_emb_TSNE30_1000,
                     _labels=Y_code,
                     considered_lb=considered_lb,
                     all_labels=all_novel_labels,
                     n_shot=0,
                     n_aug=0,
                     title='class codes',
                     save_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'class_code.png'))
    if len(training_results) > 6:
        X_code_pose = None
        for lb in lb_for_dim_reduction:
            idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
            pose_code_for_this_lb = novel_code_pose_all[idx_for_this_lb,:]
            if X_code_pose is None:
                X_code_pose = pose_code_for_this_lb
            else:
                X_code_pose = np.concatenate((X_code_pose, pose_code_for_this_lb), axis=0)
        X_code_pose_emb_TSNE30_1000 = dim_reduction(X_code_pose, 'TSNE', 2, 30, 1000)
        plot_emb_results(_emb=X_code_pose_emb_TSNE30_1000,
                         _labels=Y_code,
                         considered_lb=considered_lb,
                         all_labels=all_novel_labels,
                         n_shot=0,
                         n_aug=0,
                         title='appearance codes',
                         save_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'pose_code.png'))
    ## (8) Visualize logits of hallucinated features using the linear classifier trained on real (base + novel) features with many shots per class
    # tf.reset_default_graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     net = MSL(sess,
    #               model_name='_MSL_extPT_lr3e3_ite_0',
    #               result_path=os.path.join(args.result_path, args.hallucinator_name), # not important if we only call net.get_hal_logits() with gen_from correctly specified
    #               fc_dim=args.fc_dim,
    #               n_class=args.n_class,
    #               n_base_class=args.n_base_class,
    #               l2scale=args.l2scale)
    #     net.build_model()
    #     hal_logits_dict = net.get_hal_logits(final_novel_feat_dict=final_novel_feat_dict,
    #                                          n_shot=args.n_shot,
    #                                          n_aug=args.n_aug,
    #                                          gen_from=os.path.join(args.result_path, 'msl', '_MSL_extPT_lr3e3_ite_0', 'models'))
    #     print('hal_logits_dict.keys():', hal_logits_dict.keys())
    #     all_classes = sorted(set(labels_all_train))
    #     is_all = np.array([(i in all_classes) for i in range(args.n_class)], dtype=int)
    #     for lb in lb_for_dim_reduction:
    #         hal_logits = hal_logits_dict[lb]
    #         score = np.exp(hal_logits) / np.repeat(np.sum(np.exp(hal_logits), axis=1, keepdims=True), repeats=args.n_class, axis=1)
    #         score_all = score * is_all
    #         best_n = np.argsort(score_all, axis=1)[:,-args.n_top:]
    #         # print('for label %d, best_n.shape: %s' % (lb, best_n.shape))
    #         print('for label %d, the top %d scores of the first %d hallucinated features are:' % (lb, args.n_top, n_hal_plot))
    #         for idx in range(n_hal_plot):
    #             print('hal %d (DFHN label: %s), ' % (idx, corresponding_lb_DFHN_dict[lb][idx]), end='')
    #             print(best_n[idx, :])
    ## (9) Visualize logits of hallucinated features using the linear classifier (and a prototypical network)
    ##     trained on real (base + novel) features with many shots per class
    # tf.reset_default_graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     net = MSL_PN(sess,
    #                  model_name='_MSL_PN_extPT_lr3e3_ite_0',
    #                  result_path=os.path.join(args.result_path, args.hallucinator_name), # not important if we only call net.get_hal_logits() with gen_from correctly specified
    #                  fc_dim=args.fc_dim,
    #                  n_class=args.n_class,
    #                  n_base_class=args.n_base_class,
    #                  l2scale=args.l2scale,
    #                  with_BN=True)
    #     net.build_model()
    #     hal_logits_dict = net.get_hal_logits(final_novel_feat_dict=final_novel_feat_dict,
    #                                          n_shot=args.n_shot,
    #                                          n_aug=args.n_aug,
    #                                          gen_from=os.path.join(args.result_path, 'HAL_PN_only_withBN_m20n5q40_ep15_extPT_lr1e4', '_MSL_PN_extPT_lr3e3_ite_0', 'models'))
    #     print('hal_logits_dict.keys():', hal_logits_dict.keys())
    #     all_classes = sorted(set(labels_all_train))
    #     is_all = np.array([(i in all_classes) for i in range(args.n_class)], dtype=int)
    #     for lb in lb_for_dim_reduction:
    #         hal_logits = hal_logits_dict[lb]
    #         score = np.exp(hal_logits) / np.repeat(np.sum(np.exp(hal_logits), axis=1, keepdims=True), repeats=args.n_class, axis=1)
    #         score_all = score * is_all
    #         best_n = np.argsort(score_all, axis=1)[:,-args.n_top:]
    #         # print('for label %d, best_n.shape: %s' % (lb, best_n.shape))
    #         print('for label %d, the top %d scores of the first %d hallucinated features are:' % (lb, args.n_top, n_hal_plot))
    #         for idx in range(n_hal_plot):
    #             print('hal %d (DFHN label: %s), ' % (idx, corresponding_lb_DFHN_dict[lb][idx]), end='')
    #             print(best_n[idx, :])

    ## (10) Print logits of hal features using the current linear_cls (and proto_enc)
    if args.GAN or args.GAN2 or args.AFHN or args.DFHN:
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_frac)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if args.GAN:
                net = FSL_PN_GAN(sess,
                                 model_name=args.model_name,
                                 result_path=os.path.join(args.result_path, args.hallucinator_name),
                                 fc_dim=args.fc_dim,
                                 n_class=args.n_class,
                                 n_base_class=args.n_base_class,
                                 l2scale=args.l2scale,
                                 z_dim=args.z_dim,
                                 z_std=args.z_std)
            elif args.GAN2:
                net = FSL_PN_GAN2(sess,
                                  model_name=args.model_name,
                                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                                  fc_dim=args.fc_dim,
                                  n_class=args.n_class,
                                  n_base_class=args.n_base_class,
                                  l2scale=args.l2scale,
                                  z_dim=args.z_dim,
                                  z_std=args.z_std)
            elif args.AFHN:
                net = FSL_PN_AFHN(sess,
                                  model_name=args.model_name,
                                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                                  fc_dim=args.fc_dim,
                                  n_class=args.n_class,
                                  n_base_class=args.n_base_class,
                                  l2scale=args.l2scale,
                                  z_dim=args.z_dim,
                                  z_std=args.z_std)
            elif args.DFHN:
                net = FSL_PN_DFHN(sess,
                                  model_name=args.model_name,
                                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                                  fc_dim=args.fc_dim,
                                  n_class=args.n_class,
                                  n_base_class=args.n_base_class,
                                  l2scale=args.l2scale,
                                  n_gallery_per_class=args.n_gallery_per_class,
                                  n_base_lb_per_novel=args.n_base_lb_per_novel,
                                  use_canonical_gallery=args.use_canonical_gallery,
                                  n_clusters_per_class=args.n_clusters_per_class)
            net.build_model()
            hal_logits_dict = net.get_hal_logits(final_novel_feat_dict=final_novel_feat_dict,
                                                 n_shot=args.n_shot,
                                                 n_aug=args.n_aug,
                                                 gen_from=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'models'))
            print('hal_logits_dict.keys():', hal_logits_dict.keys())
            all_classes = sorted(set(labels_all_train))
            is_all = np.array([(i in all_classes) for i in range(args.n_class)], dtype=int)
            for lb in lb_for_dim_reduction:
                hal_logits = hal_logits_dict[lb]
                score = np.exp(hal_logits) / np.repeat(np.sum(np.exp(hal_logits), axis=1, keepdims=True), repeats=args.n_class, axis=1)
                score_all = score * is_all
                # print('score_all.shape:', score_all.shape)
                best_n = np.argsort(score_all, axis=1)[:,-args.n_top:]
                # print('for label %d, best_n.shape: %s' % (lb, best_n.shape))
                print('for label %d, the top %d scores of the first %d hallucinated features are:' % (lb, args.n_top, n_hal_plot))
                if args.DFHN:
                    for idx in range(n_hal_plot):
                        print('hal %d (DFHN label: %s): ' % (idx, corresponding_lb_DFHN_dict[lb][idx]), end='')
                        print(best_n[idx, :], end=', ')
                        print('score entropy: %.4f' % entropy(list(score_all[idx,:])))
                else:
                    for idx in range(n_hal_plot):
                        print('hal %d: ' % idx, end='')
                        print(best_n[idx, :], end=', ')
                        print('score entropy: %.4f' % entropy(list(score_all[idx,:])))
                averaged_entropy = np.mean([entropy(list(score_all[i,:])) for i in range(score_all.shape[0])])
                print('averaged_entropy:', averaged_entropy)
                        
    
if __name__ == '__main__':
    main()
