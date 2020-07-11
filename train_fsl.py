import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_fsl import FSL
from model_fsl import FSL_PN_GAN, FSL_PN_GAN2
from model_fsl import FSL_PN_PoseRef
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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--model_name', type=str, help='Folder name to save FSL models and learning curves')
    parser.add_argument('--extractor_folder', type=str, help='Folder name of the saved feature extracted by the pre-trained backbone')
    parser.add_argument('--hallucinator_name', default='Baseline', type=str, help='Folder name of the saved hallucinator model (default Baseline: no hallucination)')
    parser.add_argument('--hal_epoch', default=0, type=int, help='Hallucinator version (number of epoch), default 0: use the latest version (None)')
    parser.add_argument('--image_path', default='/data/put_data/cclin/datasets/ILSVRC2012/', type=str, help='Path of the raw images (for visualization)')
    parser.add_argument('--n_class', default=100, type=int, help='Number of all classes (base + validation or base + novel)')
    parser.add_argument('--n_base_class', default=100, type=int, help='Number of base classes')
    parser.add_argument('--n_shot', default=1, type=int, help='Number of shot')
    parser.add_argument('--n_aug', default=40, type=int, help='Number of samples per training class AFTER hallucination')
    parser.add_argument('--n_top', default=5, type=int, help='Number to compute the top-n accuracy')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size for training the final linear classifier')
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate for training the final linear classifier')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale for training the final linear classifier')
    parser.add_argument('--num_ite', default=10000, type=int, help='Number of iterations for training the final linear classifier')
    parser.add_argument('--fc_dim', default=512, type=int, help='Feature dimension')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    parser.add_argument('--label_key', default='image_labels', type=str, help='image_labels or image_labels_id')
    parser.add_argument('--exp_tag', type=str, help='cv or final for imagenet-1k; val or novel for other datasets')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN-based hallucinator (with one more layer in the hallucinator) if present')
    parser.add_argument('--PoseRef', action='store_true', help='Use PoseRef-based hallucinator if present')
    parser.add_argument('--AFHN', action='store_true', help='Use AFHN-based hallucinator if present')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
    parser.add_argument('--z_std', default=1.0, type=float, help='Standard deviation of the input noise to the GAN-based hallucinator')
    parser.add_argument('--gpu_frac', default=0.5, type=float, help='per_process_gpu_memory_fraction (0.0~1.0)')
    parser.add_argument('--with_BN', action='store_true', help='Use batch_norm() in the feature extractor mode if present')
    parser.add_argument('--with_pro', action='store_true', help='Use additional embedding network for prototypical network if present')
    parser.add_argument('--ite_idx', default=0, type=int, help='iteration index for imagenet-1k dataset (0, 1, 2, 3, 4)')
    parser.add_argument('--n_gallery_per_class', default=0, type=int, help='Number of samples per class in the gallery set, default 0: use the whole base-class dataset')
    parser.add_argument('--n_base_lb_per_novel', default=5, type=int, help='number of base classes as the candidates of pose-ref for each novel class')
    parser.add_argument('--use_canonical_gallery', action='store_true', help='Use gallery_indexes_canonical_XX.npy if present')
    parser.add_argument('--n_clusters_per_class', default=0, type=int, help='Number of clusters per base class for making the gallery set')
    parser.add_argument('--test_mode', action='store_true', help='After training, return novel class code and pose code if present')
    
    args = parser.parse_args()
    train(args)
    inference(args)
    if args.test_mode:
        visualize(args)

# Hallucination and training
def train(args):
    print('============================ train ============================')
    if args.exp_tag == 'cv':
        train_novel_fname = 'cv_novel_train_feat'
        train_base_fname = 'cv_base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'cv_novel_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'cv_novel_idx_%d.npy' % args.ite_idx))
        else:
            novel_idx = None
        # print()
        # for i in range(5):
        #     print(novel_idx[i,:])
        # print()
    elif args.exp_tag == 'final':
        train_novel_fname = 'final_novel_train_feat'
        train_base_fname = 'final_base_train_feat'
        if os.path.exists(os.path.join(args.result_path, args.extractor_folder, 'final_novel_idx_%d.npy' % args.ite_idx)):
            novel_idx = np.load(os.path.join(args.result_path, args.extractor_folder, 'final_novel_idx_%d.npy' % args.ite_idx))
        else:
            novel_idx = None
        # print()
        # for i in range(5):
        #     print(novel_idx[i,:])
        # print()
    elif args.exp_tag == 'val':
        train_novel_fname = 'val_train_feat'
        train_base_fname = 'base_train_feat'
        novel_idx = None
    elif args.exp_tag == 'novel':
        train_novel_fname = 'novel_train_feat'
        train_base_fname = 'base_train_feat'
        novel_idx = None
    
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
                             z_std=args.z_std,
                             with_BN=args.with_BN,
                             with_pro=args.with_pro)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro)
        elif args.PoseRef:
            net = FSL_PN_PoseRef(sess,
                                 model_name=args.model_name,
                                 result_path=os.path.join(args.result_path, args.hallucinator_name),
                                 fc_dim=args.fc_dim,
                                 n_class=args.n_class,
                                 n_base_class=args.n_base_class,
                                 l2scale=args.l2scale,
                                 n_gallery_per_class=args.n_gallery_per_class,
                                 n_base_lb_per_novel=args.n_base_lb_per_novel,
                                 with_BN=args.with_BN,
                                 with_pro=args.with_pro,
                                 use_canonical_gallery=args.use_canonical_gallery,
                                 n_clusters_per_class=args.n_clusters_per_class)
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
                             z_std=args.z_std,
                             with_BN=args.with_BN,
                             with_pro=args.with_pro)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                              model_name=args.model_name,
                              result_path=os.path.join(args.result_path, args.hallucinator_name),
                              fc_dim=args.fc_dim,
                              n_class=args.n_class,
                              n_base_class=args.n_base_class,
                              l2scale=args.l2scale,
                              z_dim=args.z_dim,
                              z_std=args.z_std,
                              with_BN=args.with_BN,
                              with_pro=args.with_pro)
        elif args.PoseRef:
            net = FSL_PN_PoseRef(sess,
                                 model_name=args.model_name,
                                 result_path=os.path.join(args.result_path, args.hallucinator_name),
                                 fc_dim=args.fc_dim,
                                 n_class=args.n_class,
                                 n_base_class=args.n_base_class,
                                 l2scale=args.l2scale,
                                 n_gallery_per_class=args.n_gallery_per_class,
                                 n_base_lb_per_novel=args.n_base_lb_per_novel,
                                 with_BN=args.with_BN,
                                 with_pro=args.with_pro,
                                 use_canonical_gallery=args.use_canonical_gallery,
                                 n_clusters_per_class=args.n_clusters_per_class)
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
    fig = plt.figure(figsize=(n_col*2, n_row*2))
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.05, hspace=0.05)
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
                            alpha=0.2,
                            marker='x',
                            s=30)
            #### plot hallucinated features as triangles
            if plot_hal:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_aug, 0],
                                          y=X_embedded_lb[n_shot:n_aug, 1],
                                          alpha=0.5,
                                          marker='^',
                                          color=color_list[color_idx],
                                          s=60))
            else:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_aug, 0],
                                          y=X_embedded_lb[n_shot:n_aug, 1],
                                          alpha=0.5,
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
    ## (2) Load seed and hallucinated features
    training_results = np.load(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'), allow_pickle=True)
    features_novel_final_dict = training_results[2]
    pose_feat_dict = training_results[3]
    ## (3) Combine seed/hal features and real features
    # allow dimension reduction using only a subset of classes
    # lb_for_dim_reduction = all_novel_labels
    # lb_for_dim_reduction = [166, 50, 100, 152, 176]
    lb_for_dim_reduction = np.random.choice(all_novel_labels, 5, replace=False)
    X_all = None
    Y_all = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        feat_for_this_lb = np.concatenate((features_novel_final_dict[lb], features_novel_train[idx_for_this_lb,:]), axis=0)
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
    ## (5) Find the closest real feature (and the corresponding image) for each hallucinated feature
    for lb_idx in range(5):
        considered_lb = lb_for_dim_reduction[lb_idx]
        considered_feat_seed = features_novel_final_dict[considered_lb][:args.n_shot,:]
        considered_feat_hal = features_novel_final_dict[considered_lb][args.n_shot:,:]
        considered_feat_pose = pose_feat_dict[considered_lb]

        nearest_idx_seed, nearest_diff_seed = find_nearest_idx(features_novel_train, considered_feat_seed[0])
        nearest_indexes_hal = []
        nearest_differences_hal = []
        nearest_indexes_pose = []
        nearest_differences_pose = []
        n_hal_plot = min(8, args.n_aug-args.n_shot)
        for idx in range(n_hal_plot):
            nearest_idx, nearest_diff = find_nearest_idx(features_novel_train, considered_feat_hal[idx])
            nearest_indexes_hal.append(nearest_idx)
            nearest_differences_hal.append(nearest_diff)
            nearest_idx, nearest_diff = find_nearest_idx(features_base_train, considered_feat_pose[idx])
            nearest_indexes_pose.append(nearest_idx)
            nearest_differences_pose.append(nearest_diff)

        x_dim = 84
        img_array = np.empty((3*n_hal_plot, x_dim, x_dim, 3), dtype='uint8')
        subtitles = []
        for i in range(len(nearest_indexes_hal)):
            ## (1) put seed image in the 1st row
            file_path = os.path.join(args.image_path, fnames_novel_train[nearest_idx_seed])
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array[i,:] = img
            ## (2) put pose-ref image in the 2nd row
            idx = nearest_indexes_pose[i]
            file_path = os.path.join(args.image_path, fnames_base_train[idx])
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array[i+len(nearest_indexes_hal),:] = img
            ## (3) put hal image in the 3rd row
            idx = nearest_indexes_hal[i]
            file_path = os.path.join(args.image_path, fnames_novel_train[idx])
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array[i+2*len(nearest_indexes_hal),:] = img
            subtitles.append(str(labels_novel_train[idx]))
        # print(subtitles)
        fig = plot(img_array, 3, n_hal_plot, x_dim=x_dim)
        plt.savefig(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'nearest_images_%3d.png' % considered_lb), bbox_inches='tight')
    ## (6) Visualize class codes and pose codes
    novel_code_class_all = training_results[5]
    novel_code_pose_all = training_results[6]
    X_code_class = None
    X_code_pose = None
    Y_code = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        class_code_for_this_lb = novel_code_class_all[idx_for_this_lb,:]
        pose_code_for_this_lb = novel_code_pose_all[idx_for_this_lb,:]
        labels_for_this_lb = np.repeat(lb, repeats=len(idx_for_this_lb))
        if X_code_class is None:
            X_code_class = class_code_for_this_lb
            X_code_pose = pose_code_for_this_lb
            Y_code = labels_for_this_lb
        else:
            X_code_class = np.concatenate((X_code_class, class_code_for_this_lb), axis=0)
            X_code_pose = np.concatenate((X_code_pose, pose_code_for_this_lb), axis=0)
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
    X_code_pose_emb_TSNE30_1000 = dim_reduction(X_code_pose, 'TSNE', 2, 30, 1000)
    plot_emb_results(_emb=X_code_pose_emb_TSNE30_1000,
                     _labels=Y_code,
                     considered_lb=considered_lb,
                     all_labels=all_novel_labels,
                     n_shot=0,
                     n_aug=0,
                     title='appearance codes',
                     save_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'pose_code.png'))
    ## (7) How about the prototypical-network-embeded features?
    features_novel_embeded_dict = training_results[4]
    X_all = None
    Y_all = None
    for lb in lb_for_dim_reduction:
        idx_for_this_lb = [i for i in range(len(labels_novel_train)) if labels_novel_train[i] == lb]
        feat_for_this_lb = np.concatenate((features_novel_embeded_dict[lb], novel_code_class_all[idx_for_this_lb,:]), axis=0)
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

if __name__ == '__main__':
    main()