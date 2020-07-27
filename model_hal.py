import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.contrib.layers import l2_regularizer
import tqdm

from ops import *
from utils import *

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def dim_reduction(_input,
                  method='TSNE',
                  n_components=2,
                  perplexity=30,
                  n_iter=1000):
    if method == 'TSNE':
        return TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter).fit_transform(_input)
    elif method == 'PCA':
        return PCA(n_components=n_components).fit_transform(_input)

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

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

def plot_emb_results(_emb, # 2-dim feature 
                     _labels, # 1-dim label list of _emb (e.g., absolute novel labeling)
                     considered_lb,
                     all_labels,
                     n_shot,
                     n_min,
                     color_list=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
                     plot_seed=True,
                     plot_real=True,
                     plot_hal=True,
                     alpha_for_seed=0.7,
                     alpha_for_hal=0.5,
                     alpha_for_real=0.3,
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
                            alpha=alpha_for_seed, ##### default: 0.7
                            s=200)
            #### plot all real features as crosses
            if plot_real:
                plt.scatter(x=X_embedded_lb[n_min:n_feat_per_lb, 0],
                            y=X_embedded_lb[n_min:n_feat_per_lb, 1],
                            color=color_list[color_idx],
                            alpha=alpha_for_real,  ##### default: 0.3
                            marker='x',
                            s=30)
            #### plot hallucinated features as triangles
            if plot_hal:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_min, 0],
                                          y=X_embedded_lb[n_shot:n_min, 1],
                                          alpha=alpha_for_hal, ##### default: 0.5
                                          marker='^',
                                          color=color_list[color_idx],
                                          s=60))
            else:
                plt_lb.append(plt.scatter(x=X_embedded_lb[n_shot:n_min, 0],
                                          y=X_embedded_lb[n_shot:n_min, 1],
                                          alpha=alpha_for_hal, ##### default: 0.5
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
                    plt.scatter(x=X_embedded_lb[n_min:n_feat_per_lb, 0],
                                y=X_embedded_lb[n_min:n_feat_per_lb, 1],
                                color='white',
                                marker='x',
                                s=30)
                plt.scatter(x=X_embedded_lb[n_shot:n_min, 0],
                            y=X_embedded_lb[n_shot:n_min, 1],
                            marker='^',
                            color='white',
                            s=30)
    plt.title(title, fontsize=20)
    plt.legend(plt_lb,
               ['label %d' % lb for lb in all_labels if lb in considered_lb],
               fontsize=16, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

class HAL_PN_GAN(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 label_key='image_labels',
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=512,
                 z_std=1.0,
                 l2scale=0.001,
                 n_train_class=64,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=4):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_aug = n_aug
        self.n_query_all = n_query_all
        self.fc_dim = fc_dim
        self.z_dim = z_dim
        self.z_std = z_std
        self.l2scale = l2scale
        self.n_train_class = n_train_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.num_parallel_calls = num_parallel_calls
        
        ### prepare datasets
        self.train_path = train_path
        self.label_key = label_key
        self.train_base_dict = unpickle(self.train_path)
        self.train_feat_list = self.train_base_dict['features']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        self.train_class_list_raw = self.train_base_dict[self.label_key]
        all_train_class = sorted(set(self.train_class_list_raw))
        print('original train class labeling:')
        print(all_train_class)
        label_mapping = {}
        for new_lb in range(self.n_train_class):
            label_mapping[all_train_class[new_lb]] = new_lb
        self.train_label_list = np.array([label_mapping[old_lb] for old_lb in self.train_class_list_raw])
        print('new train class labeling:')
        print(sorted(set(self.train_label_list)))
        
        ### [2020/03/21] make candidate indexes for each label
        self.candidate_indexes_each_lb_train = {}
        for lb in range(self.n_train_class):
            self.candidate_indexes_each_lb_train[lb] = [idx for idx in range(len(self.train_label_list)) if self.train_label_list[idx] == lb]
        self.all_train_labels = set(self.train_label_list)
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train_hal = tf.placeholder('bool', name='bn_train_hal')
        
        ### (1) episodic training
        ### dataset
        self.support_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_shot, self.fc_dim], name='support_features')
        self.query_features = tf.placeholder(tf.float32, shape=[self.n_query_all, self.fc_dim], name='query_features')
        self.query_labels = tf.placeholder(tf.int32, shape=[self.n_query_all], name='query_labels')
        self.support_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='support_labels')
        self.support_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='support_classes')

        ### basic operation
        self.support_feat_flat = tf.reshape(self.support_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### hallucination flow
        self.hal_feat = self.build_augmentor(self.support_features, self.n_way, self.n_shot, self.n_aug) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.hallucinated_features = tf.reshape(self.hal_feat, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_features, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
            self.hal_class_code = self.proto_encoder(self.hal_feat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.support_class_code = self.support_feat_flat
            self.query_class_code = self.query_features
            self.hal_class_code = self.hal_feat
        ### prototypical network data flow using the augmented support set
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                                   logits=self.logits_pro_aug,
                                                                                   name='loss_pro_aug'))
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_pro_aug+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
        
        ### model saver (keep the best checkpoint)
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)

        ### Count number of trainable variables
        total_params = 0
        for var in self.trainable_vars:
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total_params += var_params
        print('total number of parameters: %d' % total_params)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## "For PN, the embedding architecture consists of two MLP layers with ReLU as the activation function." (Y-X Wang, 2018)
    def proto_encoder(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('pro', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
        return x
    
    ## "For our hallucinator G, we use a three layer MLP with ReLU as the activation function." (Y-X Wang, 2018)
    ## Use linear_identity() that initializes network parameters with identity matrix
    def hallucinator(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense2') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn2')
            x = tf.nn.relu(x, name='relu2')
            x = linear_identity(x, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu3')
        return x
    
    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, support, n_way, n_shot, n_aug, reuse=False):
        input_z_vec = tf.random_normal([n_way*(n_aug-n_shot), self.z_dim], stddev=self.z_std)
        ### Randomly select (n_aug-n_shot) samples per class as seeds
        ### [2019/12/18] Use tg.gather_nd() to collect randomly selected samples from support set
        ### Ref: https://github.com/tensorflow/docs/blob/r1.4/site/en/api_docs/api_docs/python/tf/gather_nd.md
        ### For N-way, we want something like: [[[0, idx_1], [0, idx_2], ..., [0, idx_a]],
        ###                                     [[1, idx_1], [1, idx_2], ..., [1, idx_a]],
        ###                                     ...,
        ###                                     [[N-1, idx_1], [N-1, idx_2], ..., [N-1, idx_a]]],
        ### where idx_1, idx_2, ..., idx_a are the sampled indexes for each class (each within the range {0, 1, ..., n_shot-1}),
        ### and a=n_aug-n_shot is the number of additional samples we want for each class.
        idxs_for_each_class = np.array([np.random.choice(n_shot, n_aug-n_shot) for label_b in range(n_way)]) ### E.g., for 3-way 2-shot with n_aug=4, sampled [[1, 1], [1, 0], [1, 1]]
        idxs_for_each_class = idxs_for_each_class.reshape([n_way*(n_aug-n_shot), 1]) ### reshape into [[1], [1], [1], [0], [1], [1]] for later combination with repeated class indexes
        repeated_class_id = np.expand_dims(np.repeat(np.arange(n_way), n_aug-n_shot), 1) ### E.g., for 3-way 2-shot with n_aug=4, we want [[0], [0], [1], [1], [2], [2]]
        idxs_for_gathering = np.concatenate((repeated_class_id, idxs_for_each_class), axis=1).reshape([n_way, n_aug-n_shot, 2]) ### E.g., for 3-way 2-shot with n_aug=4, we want [[[0, 1], [0, 1]], [[1, 1], [1, 0]], [[2, 1], [2, 1]]]
        idxs_for_gathering_tensor = tf.convert_to_tensor(idxs_for_gathering)
        sampled_support = tf.gather_nd(support, idxs_for_gathering_tensor)
        sampled_support = tf.reshape(sampled_support, shape=[-1, self.fc_dim]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([sampled_support, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat
    
    def train(self,
              num_epoch=100,
              n_ite_per_epoch=600,
              lr_start=1e-5,
              lr_decay=0.5,
              lr_decay_step=20,
              patience=10):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     print(m.values())
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### episodic training
        loss_train = []
        acc_train = []
        start_time = time.time()
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        for epoch in range(1, (num_epoch+1)):
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                ##### make episode
                skip_this_episode = False
                selected_lbs = np.random.choice(list(self.all_train_labels), self.n_way, replace=False)
                try:
                    selected_indexes = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs[lb_idx]], self.n_shot+self.n_query_all//self.n_way, replace=False)) \
                                        for lb_idx in range(self.n_way)]
                except:
                    print('[Training] Skip this episode since there are not enough samples for some label')
                    skip_this_episode = True
                if skip_this_episode:
                    continue
                support_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][0:self.n_shot]] for lb_idx in range(self.n_way)])
                support_features = np.reshape(support_features, (self.n_way, self.n_shot, self.fc_dim))
                query_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][self.n_shot:]] for lb_idx in range(self.n_way)])
                query_labels = np.concatenate([np.repeat(lb_idx, self.n_query_all//self.n_way) for lb_idx in range(self.n_way)])
                support_labels = selected_lbs
                support_classes = np.array([self.train_class_list_raw[selected_indexes[i][0]] for i in range(self.n_way)])
                _, loss, acc = self.sess.run([self.opt, self.loss_pro_aug, self.acc_pro_aug],
                                             feed_dict={self.support_features: support_features,
                                                        self.query_features: query_features,
                                                        self.query_labels: query_labels,
                                                        self.support_labels: support_labels,
                                                        self.support_classes: support_classes,
                                                        self.bn_train_hal: True,
                                                        self.learning_rate: lr})
                loss_ite_train.append(loss)
                acc_ite_train.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training accuracy: %f' % \
                (epoch, lr, np.mean(loss_ite_train), np.mean(acc_ite_train)))
                
        #### save model
        self.saver_hal_pro.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
                                global_step=epoch)
        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, acc_train]

class HAL_PN_GAN2(HAL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 label_key='image_labels',
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=512,
                 z_std=1.0,
                 l2scale=0.001,
                 n_train_class=64,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=4):
        super(HAL_PN_GAN2, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          train_path,
                                          label_key,
                                          n_way,
                                          n_shot,
                                          n_aug,
                                          n_query_all,
                                          fc_dim,
                                          z_dim,
                                          z_std,
                                          l2scale,
                                          n_train_class,
                                          with_BN,
                                          with_pro,
                                          bnDecay,
                                          epsilon,
                                          num_parallel_calls)
    
    def hallucinator(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn1')
            x = tf.nn.relu(x, name='relu1')
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense2') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn2')
            x = tf.nn.relu(x, name='relu2')
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense3') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn3')
            x = tf.nn.relu(x, name='relu3')
            x = linear_identity(x, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu4')
        return x

class HAL_PN_AFHN(HAL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 label_key='image_labels',
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=512,
                 z_std=1.0,
                 l2scale=0.001,
                 n_train_class=64,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=4,
                 lambda_meta=1.0,
                 lambda_tf=0.0,
                 lambda_ar=0.0,
                 gp_scale=10.0,
                 d_per_g=5):
        super(HAL_PN_AFHN, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          train_path,
                                          label_key,
                                          n_way,
                                          n_shot,
                                          n_aug,
                                          n_query_all,
                                          fc_dim,
                                          z_dim,
                                          z_std,
                                          l2scale,
                                          n_train_class,
                                          with_BN,
                                          with_pro,
                                          bnDecay,
                                          epsilon,
                                          num_parallel_calls)
        self.lambda_meta = lambda_meta
        self.lambda_tf = lambda_tf
        self.lambda_ar = lambda_ar
        self.gp_scale = gp_scale
        self.d_per_g = d_per_g
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train_hal = tf.placeholder('bool', name='bn_train_hal')
        
        ### (1) episodic training
        ### dataset
        self.support_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_shot, self.fc_dim], name='support_features')
        self.query_features = tf.placeholder(tf.float32, shape=[self.n_query_all, self.fc_dim], name='query_features')
        self.query_labels = tf.placeholder(tf.int32, shape=[self.n_query_all], name='query_labels')
        self.support_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='support_labels')
        self.support_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='support_classes')
        
        ### basic operation
        self.support_feat_flat = tf.reshape(self.support_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### hallucination flow
        self.support_ave = tf.reduce_mean(self.support_features, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        ### [NOTE] make 2 hallucination for each class in the support set
        self.hal_feat_1, self.z_1 = self.build_augmentor(self.support_ave, self.n_way, self.n_shot, self.n_shot+1) ### shape: [self.n_way, self.fc_dim], [self.n_way, self.z_dim]
        self.hallucinated_features_1 = tf.reshape(self.hal_feat_1, shape=[self.n_way, 1, -1]) ### shape: [self.n_way, 1, self.fc_dim]
        self.hal_feat_2, self.z_2 = self.build_augmentor(self.support_ave, self.n_way, self.n_shot, self.n_shot+1, reuse=True) ### shape: [self.n_way, self.fc_dim], [self.n_way, self.z_dim]
        self.hallucinated_features_2 = tf.reshape(self.hal_feat_2, shape=[self.n_way, 1, -1]) ### shape: [self.n_way, 1, self.fc_dim]
        self.hallucinated_features = tf.concat((self.hallucinated_features_1, self.hallucinated_features_2), axis=1) ### shape: [self.n_way, 2, self.fc_dim]
        self.hal_feat = tf.reshape(self.hallucinated_features, shape=[self.n_way*2, -1]) ### shape: [self.n_way*2, self.fc_dim]
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_features, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
            self.hal_class_code = self.proto_encoder(self.hal_feat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.support_class_code = self.support_feat_flat
            self.query_class_code = self.query_features
            self.hal_class_code = self.hal_feat
        ### [2020/06/24] Follow AFHN (K. Li, CVPR 2020) to use cosine similarity between hallucinated and query features to compute classification error
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, 2, -1]) ### shape: [self.n_way, 2, self.fc_dim]
        squared_a = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.hal_encode), axis=2, keep_dims=True)), multiples=[1, 1, self.fc_dim])
        normalize_a = self.hal_encode / (squared_a + 1e-10)
        normalize_a_tile = tf.tile(tf.transpose(normalize_a, perm=[1, 0, 2]), multiples=[self.n_query_all, 1, 1])
        query_labels_repeat = tf.reshape(tf.tile(tf.expand_dims(self.query_labels, axis=1), multiples=[1, 2]), [-1])
        query_labels_repeat_vec = tf.one_hot(query_labels_repeat, self.n_way)
        squared_b = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.query_class_code), axis=1, keep_dims=True)), multiples=[1, self.fc_dim])
        normalize_b = self.query_class_code / (squared_b + 1e-10)
        normalize_b_repeat = tf.reshape(tf.tile(tf.expand_dims(normalize_b, axis=1), multiples=[1, self.n_way*2, 1]), [self.n_query_all*2, self.n_way, -1])
        cos_sim_ab_exp = tf.exp(tf.reduce_sum(normalize_a_tile * normalize_b_repeat, axis=2))
        p_y = cos_sim_ab_exp / (tf.reduce_sum(cos_sim_ab_exp, axis=1, keep_dims=True) + 1e-10)
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=query_labels_repeat_vec,
                                                                                   logits=cos_sim_ab_exp,
                                                                                   name='loss_pro_aug'))
        ### prototypical network data flow using the augmented support set
        # self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        # self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        # self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        # self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        # self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        # self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
        #                                                                            logits=self.logits_pro_aug,
        #                                                                            name='loss_pro_aug'))
        # self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)
        ### still use prototypical network data flow with the augmented support set to compute accuracy
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)
        
        ### Anti-collapse regularizer in AFHN (K. Li, CVPR 2020)
        squared_z_1 = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.z_1), axis=1, keep_dims=True)), multiples=[1,self.z_dim])
        normalize_z_1 = self.z_1 / (squared_z_1 + 1e-10)
        squared_z_2 = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.z_2), axis=1, keep_dims=True)), multiples=[1,self.z_dim])
        normalize_z_2 = self.z_2 / (squared_z_2 + 1e-10)
        self.cos_sim_z1z2 = tf.reduce_sum(normalize_z_1 * normalize_z_2, axis=1)
        squared_s_1 = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.hal_feat_1), axis=1, keep_dims=True)), multiples=[1,self.fc_dim])
        normalize_s_1 = self.hal_feat_1 / (squared_s_1 + 1e-10)
        squared_s_2 = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.hal_feat_2), axis=1, keep_dims=True)), multiples=[1,self.fc_dim])
        normalize_s_2 = self.hal_feat_2 / (squared_s_2 + 1e-10)
        self.cos_sim_s1s2 = tf.reduce_sum(normalize_s_1 * normalize_s_2, axis=1)
        self.loss_anti_collapse = 1 / (tf.reduce_mean((1 - self.cos_sim_s1s2) / (1 - self.cos_sim_z1z2 + 1e-10)) + 1e-10)

        ### [2020/06/04] Follow AFHN (K. Li, CVPR 2020) to use a discriminator to make the hallucinated features (self.hal_feat) more realistic
        self.s_real = tf.reshape(self.support_ave, shape=[-1, self.fc_dim]) ### shape: [self.n_way, self.fc_dim]
        # self.feat_real_1 = tf.concat((self.s_real, self.z_1), axis=1) ### shape: [self.n_way, self.fc_dim+self.z_dim]
        # self.feat_real_2 = tf.concat((self.s_real, self.z_2), axis=1) ### shape: [self.n_way, self.fc_dim+self.z_dim]
        # self.feat_fake_1 = tf.concat((self.hal_feat_1, self.z_1), axis=1) ### shape: [self.n_way, self.fc_dim+self.z_dim]
        # self.feat_fake_2 = tf.concat((self.hal_feat_2, self.z_2), axis=1) ### shape: [self.n_way, self.fc_dim+self.z_dim]
        ### [2020/07/01] it is also very weird to include noise as the input to the discriminator
        self.feat_real_1 = self.s_real ### shape: [self.n_way, self.fc_dim]
        self.feat_real_2 = self.s_real ### shape: [self.n_way, self.fc_dim]
        self.feat_fake_1 = self.hal_feat_1 ### shape: [self.n_way, self.fc_dim]
        self.feat_fake_2 = self.hal_feat_2 ### shape: [self.n_way, self.fc_dim]
        if self.with_pro:
            self.d_logit_real_1 = self.discriminator_tf(self.proto_encoder(self.feat_real_1, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train_hal)
            self.d_logit_real_2 = self.discriminator_tf(self.proto_encoder(self.feat_real_2, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train_hal, reuse=True)
            self.d_logit_fake_1 = self.discriminator_tf(self.proto_encoder(self.feat_fake_1, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train_hal, reuse=True)
            self.d_logit_fake_2 = self.discriminator_tf(self.proto_encoder(self.feat_fake_2, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train_hal, reuse=True)
        else:
            self.d_logit_real_1 = self.discriminator_tf(self.feat_real_1, bn_train=self.bn_train_hal)
            self.d_logit_real_2 = self.discriminator_tf(self.feat_real_2, bn_train=self.bn_train_hal, reuse=True)
            self.d_logit_fake_1 = self.discriminator_tf(self.feat_fake_1, bn_train=self.bn_train_hal, reuse=True)
            self.d_logit_fake_2 = self.discriminator_tf(self.feat_fake_2, bn_train=self.bn_train_hal, reuse=True)
        ### (WGAN loss)
        self.loss_d_real = tf.reduce_mean(self.d_logit_real_1) + tf.reduce_mean(self.d_logit_real_2)
        self.loss_d_fake = tf.reduce_mean(self.d_logit_fake_1) + tf.reduce_mean(self.d_logit_fake_2)
        self.loss_d_tf = self.loss_d_fake - self.loss_d_real
        self.loss_g_tf = (-1) * (tf.reduce_mean(self.d_logit_fake_1) + tf.reduce_mean(self.d_logit_fake_2))
        epsilon_1 = tf.random_uniform([], 0.0, 1.0)
        x_hat_1 = epsilon_1 * self.s_real + (1 - epsilon_1) * self.hal_feat_1
        # d_hat_1 = self.discriminator_tf(tf.concat((x_hat_1, self.z_1), axis=1), self.bn_train_hal, reuse=True)
        d_hat_1 = self.discriminator_tf(x_hat_1, self.bn_train_hal, reuse=True)
        self.ddx_1 = tf.gradients(d_hat_1, x_hat_1)[0]
        self.ddx_1 = tf.sqrt(tf.reduce_sum(tf.square(self.ddx_1), axis=1))
        self.ddx_1 = tf.reduce_mean(tf.square(self.ddx_1 - 1.0) * self.gp_scale)
        epsilon_2 = tf.random_uniform([], 0.0, 1.0)
        x_hat_2 = epsilon_2 * self.s_real + (1 - epsilon_2) * self.hal_feat_2
        # d_hat_2 = self.discriminator_tf(tf.concat((x_hat_2, self.z_2), axis=1), self.bn_train_hal, reuse=True)
        d_hat_2 = self.discriminator_tf(x_hat_2, self.bn_train_hal, reuse=True)
        self.ddx_2 = tf.gradients(d_hat_2, x_hat_2)[0]
        self.ddx_2 = tf.sqrt(tf.reduce_sum(tf.square(self.ddx_2), axis=1))
        self.ddx_2 = tf.reduce_mean(tf.square(self.ddx_2 - 1.0) * self.gp_scale)
        self.loss_d_tf = self.loss_d_tf + self.ddx_1 + self.ddx_2
        
        ### combine all loss functions
        self.loss_d = self.lambda_tf * self.loss_d_tf
        self.loss_all = self.lambda_meta * self.loss_pro_aug + \
                        self.lambda_tf * self.loss_g_tf + \
                        self.lambda_ar * self.loss_anti_collapse

        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_d = [var for var in self.trainable_vars if ('discriminator' in var.name)]
        self.trainable_vars_hal = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_d = [reg for reg in self.all_regs if \
                            ('discriminator' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops):
            self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.5).minimize(self.loss_d+sum(self.used_regs_d),
                                                                    var_list=self.trainable_vars_d)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_all+sum(self.used_regs_hal),
                                                                  var_list=self.trainable_vars_hal)
        
        ### model saver (keep the best checkpoint)
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)

        ### Count number of trainable variables
        total_params = 0
        for var in self.trainable_vars:
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total_params += var_params
        print('total number of parameters: %d' % total_params)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## "The discriminator is also a twolayer MLP, with LeakyReLU as the activation function for the first layer
    ##  and Sigmoid for the second layer. The dimension of the hidden layer is also 1024." (K. Li, 2020)
    def discriminator_tf(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('discriminator_tf', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, 1024, add_bias=(~with_BN), name='dense1')
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = lrelu(x, name='relu1', leak=0.01)
            x = linear(x, 1, add_bias=True, name='dense2')
            #### It's weird to add sigmoid activation function for WGAN.
            # x = tf.nn.sigmoid(x, name='sigmoid1')
            return x
    
    ## "We implement the generator G as a two-layer MLP, with LeakyReLU activation for the first layer
    ##  and ReLU activation for the second one. The dimension of the hidden layer is 1024." (K. Li, CVPR 2020)
    def hallucinator(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear_identity(x, 1024, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn1')
            x = lrelu(x, name='relu1')
            x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense2') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn2')
            x = tf.nn.relu(x, name='relu2')
        return x
    
    def build_augmentor(self, support_ave, n_way, n_shot, n_aug, reuse=False):
        input_z_vec = tf.random_normal([n_way*(n_aug-n_shot), self.z_dim], stddev=self.z_std)
        support_ave_tile = tf.tile(support_ave, multiples=[1, n_aug-n_shot, 1])
        support_ave_tile_reshape = tf.reshape(support_ave_tile, shape=[-1, self.fc_dim]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([support_ave_tile_reshape, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat, input_z_vec
    
    ## [2020/07/01] test if the hallucinated features generated from the same set of noise vectors will be more and more diverse as the training proceeds
    def build_augmentor_test(self, support_ave, n_way, n_shot, n_aug, reuse=False):
        input_z_vec = tf.random_normal([n_way*(n_aug-n_shot), self.z_dim], stddev=self.z_std, seed=1002)
        support_ave_tile = tf.tile(support_ave, multiples=[1, n_aug-n_shot, 1])
        support_ave_tile_reshape = tf.reshape(support_ave_tile, shape=[-1, self.fc_dim]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([support_ave_tile_reshape, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat, input_z_vec
    
    def train(self,
              num_epoch=100,
              n_ite_per_epoch=600,
              lr_start=1e-5,
              lr_decay=0.5,
              lr_decay_step=20,
              patience=10):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     print(m.values())
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### episodic training
        loss_train = []
        acc_train = []
        start_time = time.time()
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        for epoch in range(1, (num_epoch+1)):
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                if self.lambda_tf > 0:
                    for i_d_update in range(self.d_per_g):
                        ####### make support set only
                        skip_this_episode = False
                        selected_lbs = np.random.choice(list(self.all_train_labels), self.n_way, replace=False)
                        try:
                            selected_indexes = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs[lb_idx]], self.n_shot, replace=False)) \
                                                for lb_idx in range(self.n_way)]
                        except:
                            print('[Training] Skip this episode since there are not enough samples for some label')
                            skip_this_episode = True
                        if skip_this_episode:
                            continue
                        support_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx]] for lb_idx in range(self.n_way)])
                        support_features = np.reshape(support_features, (self.n_way, self.n_shot, self.fc_dim))
                        _ = self.sess.run(self.opt_d,
                                          feed_dict={self.support_features: support_features,
                                                     self.bn_train_hal: False,
                                                     self.learning_rate: lr})
                ##### make episode
                skip_this_episode = False
                selected_lbs = np.random.choice(list(self.all_train_labels), self.n_way, replace=False)
                try:
                    selected_indexes = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs[lb_idx]], self.n_shot+self.n_query_all//self.n_way, replace=False)) \
                                        for lb_idx in range(self.n_way)]
                except:
                    print('[Training] Skip this episode since there are not enough samples for some label')
                    skip_this_episode = True
                if skip_this_episode:
                    continue
                support_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][0:self.n_shot]] for lb_idx in range(self.n_way)])
                support_features = np.reshape(support_features, (self.n_way, self.n_shot, self.fc_dim))
                query_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][self.n_shot:]] for lb_idx in range(self.n_way)])
                query_labels = np.concatenate([np.repeat(lb_idx, self.n_query_all//self.n_way) for lb_idx in range(self.n_way)])
                support_labels = selected_lbs
                support_classes = np.array([self.train_class_list_raw[selected_indexes[i][0]] for i in range(self.n_way)])
                _, loss, acc = self.sess.run([self.opt, self.loss_all, self.acc_pro_aug],
                                             feed_dict={self.support_features: support_features,
                                                        self.query_features: query_features,
                                                        self.query_labels: query_labels,
                                                        self.support_labels: support_labels,
                                                        self.support_classes: support_classes,
                                                        self.bn_train_hal: True,
                                                        self.learning_rate: lr})
                loss_ite_train.append(loss)
                acc_ite_train.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training accuracy: %f' % \
                (epoch, lr, np.mean(loss_ite_train), np.mean(acc_ite_train)))
                
        #### save model
        self.saver_hal_pro.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
                                global_step=epoch)
        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, acc_train]

class HAL_PN_PoseRef(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 val_path=None,
                 label_key='image_labels',
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 fc_dim=512,
                 l2scale=0.001,
                 n_train_class=64,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=4,
                 lambda_meta=1.0,
                 lambda_recon=1.0,
                 lambda_consistency=0.0,
                 lambda_consistency_pose=0.0,
                 lambda_intra=0.0,
                 lambda_pose_code_reg=0.0,
                 lambda_aux=0.0,
                 lambda_gan=0.0,
                 lambda_tf=0.0,
                 gp_scale=10.0,
                 d_per_g=5):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_aug = n_aug
        self.n_query_all = n_query_all
        self.n_intra = self.n_aug - self.n_shot
        self.fc_dim = fc_dim
        self.l2scale = l2scale
        self.n_train_class = n_train_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.num_parallel_calls = num_parallel_calls

        self.lambda_meta = lambda_meta
        self.lambda_recon = lambda_recon
        self.lambda_consistency = lambda_consistency
        self.lambda_consistency_pose = lambda_consistency_pose
        self.lambda_intra = lambda_intra
        self.lambda_pose_code_reg = lambda_pose_code_reg
        self.lambda_aux = lambda_aux
        self.lambda_gan = lambda_gan
        self.lambda_tf = lambda_tf
        self.gp_scale = gp_scale
        self.d_per_g = d_per_g
        
        ### prepare datasets
        self.train_path = train_path
        self.label_key = label_key
        self.train_base_dict = unpickle(self.train_path)
        self.train_feat_list = self.train_base_dict['features']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        self.train_class_list_raw = self.train_base_dict[self.label_key]
        all_train_class = sorted(set(self.train_class_list_raw))
        print('original train class labeling:')
        print(all_train_class)
        label_mapping = {}
        for new_lb in range(self.n_train_class):
            label_mapping[all_train_class[new_lb]] = new_lb
        self.train_label_list = np.array([label_mapping[old_lb] for old_lb in self.train_class_list_raw])
        print('new train class labeling:')
        print(sorted(set(self.train_label_list)))
        
        ### [2020/03/21] make candidate indexes for each label
        self.candidate_indexes_each_lb_train = {}
        for lb in range(self.n_train_class):
            self.candidate_indexes_each_lb_train[lb] = [idx for idx in range(len(self.train_label_list)) if self.train_label_list[idx] == lb]
        self.all_train_labels = set(self.train_label_list)

        # if not val_path is None:
        #     self.val_path = val_path
        #     self.val_base_dict = unpickle(self.val_path)
        #     self.val_feat_list = self.val_base_dict['features']
        #     #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #     #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        #     self.val_class_list_raw = self.val_base_dict[self.label_key]
        #     all_val_class = sorted(set(self.val_class_list_raw))
        #     print('original val class labeling:')
        #     print(all_val_class)
        #     label_mapping = {}
        #     self.n_val_class = len(all_val_class)
        #     for new_lb in range(self.n_val_class):
        #         label_mapping[all_val_class[new_lb]] = new_lb
        #     self.val_label_list = np.array([label_mapping[old_lb] for old_lb in self.val_class_list_raw])
        #     print('new val class labeling:')
        #     print(sorted(set(self.val_label_list)))
            
        #     ### [2020/03/21] make candidate indexes for each label
        #     self.candidate_indexes_each_lb_val = {}
        #     for lb in range(self.n_val_class):
        #         self.candidate_indexes_each_lb_val[lb] = [idx for idx in range(len(self.val_label_list)) if self.val_label_list[idx] == lb]
        #     self.all_val_labels = set(self.val_label_list)
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train_hal = tf.placeholder('bool', name='bn_train_hal')
        
        ### (1) episodic training
        ### dataset
        self.support_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_shot, self.fc_dim], name='support_features')
        self.query_features = tf.placeholder(tf.float32, shape=[self.n_query_all, self.fc_dim], name='query_features')
        self.query_labels = tf.placeholder(tf.int32, shape=[self.n_query_all], name='query_labels')
        self.pose_ref_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_aug-self.n_shot, self.fc_dim], name='pose_ref_features')
        self.pose_ref_intra_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_intra, self.fc_dim], name='pose_ref_intra_features')
        self.support_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='support_labels')
        self.pose_ref_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='pose_ref_labels')
        self.support_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='support_classes')
        self.pose_ref_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='pose_ref_classes')
        
        ### basic operation
        self.support_feat_flat = tf.reshape(self.support_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.pose_ref_feat_flat = tf.reshape(self.pose_ref_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref_intra_feat_flat = tf.reshape(self.pose_ref_intra_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### hallucination flow
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_features, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
            self.pose_ref_class_code = self.proto_encoder(self.pose_ref_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.support_class_code = self.support_feat_flat
            self.query_class_code = self.query_features
            self.pose_ref_class_code = self.pose_ref_feat_flat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_encode_ave = tf.reduce_mean(self.support_encode, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.support_encode_ave_tile = tf.tile(self.support_encode_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_encode_ave_tile_flat = tf.reshape(self.support_encode_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.transformed_pose_ref, self.pose_code = self.transformer(self.pose_ref_feat_flat, self.support_encode_ave_tile_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.hallucinated_features = tf.reshape(self.transformed_pose_ref, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        ### prototypical network data flow using the augmented support set
        if self.with_pro:
            self.hal_class_code = self.proto_encoder(self.transformed_pose_ref, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.hal_class_code = self.transformed_pose_ref
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                                   logits=self.logits_pro_aug,
                                                                                   name='loss_pro_aug'))
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

        ### cycle-consistency loss
        self.pose_ref_encode = tf.reshape(self.pose_ref_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.pose_ref_encode_ave = tf.reduce_mean(self.pose_ref_encode, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.pose_ref_encode_ave_tile = tf.tile(self.pose_ref_encode_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.pose_ref_encode_ave_tile_flat = tf.reshape(self.pose_ref_encode_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref_recon, self.transformed_pose_code = self.transformer(self.transformed_pose_ref, self.pose_ref_encode_ave_tile_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.loss_pose_ref_recon = tf.reduce_mean((self.pose_ref_feat_flat - self.pose_ref_recon)**2)
        self.loss_pose_code_recon = tf.reduce_mean((self.pose_code - self.transformed_pose_code)**2)
        ### reconstruction loss
        self.support_encode_ave_tile2 = tf.tile(self.support_encode_ave, multiples=[1, self.n_shot, 1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_encode_ave_tile2_flat = tf.reshape(self.support_encode_ave_tile2, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support_recon, _ = self.transformer(self.support_feat_flat, self.support_encode_ave_tile2_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True)
        self.loss_support_recon = tf.reduce_mean((self.support_feat_flat - self.support_recon)**2)
        ### intra-class transformation consistency loss
        ### (if pose-ref are of the same class as the support samples, then the transformed pose-ref should be as close to the pose-ref itself as possible)
        self.support_encode_ave_tile3 = tf.tile(self.support_encode_ave, multiples=[1, self.n_intra, 1]) ### shape: [self.n_way, self.n_intra, self.fc_dim]
        self.support_encode_ave_tile3_flat = tf.reshape(self.support_encode_ave_tile3, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_intra, self.fc_dim]
        self.transformed_pose_ref_intra, _ = self.transformer(self.pose_ref_intra_feat_flat, self.support_encode_ave_tile3_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*self.n_intra, self.fc_dim]
        self.loss_intra = tf.reduce_mean((self.pose_ref_intra_feat_flat - self.transformed_pose_ref_intra)**2)
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.loss_all = self.lambda_meta * self.loss_pro_aug + \
                        self.lambda_recon * self.loss_support_recon + \
                        self.lambda_consistency * self.loss_pose_ref_recon + \
                        self.lambda_consistency_pose * self.loss_pose_code_recon + \
                        self.lambda_intra * self.loss_intra
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_all+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
        
        ### model saver (keep the best checkpoint)
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)

        ### Count number of trainable variables
        total_params = 0
        for var in self.trainable_vars:
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total_params += var_params
        print('total number of parameters: %d' % total_params)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## pose code discriminator
    def discriminator_pose(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('discriminator_pose', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = lrelu(x, name='relu1', leak=0.01)
            x = linear(x, self.n_train_class, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            return x

    ## hallucinated feature discriminator
    def discriminator_tf(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('discriminator_tf', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = lrelu(x, name='relu1', leak=0.01)
            x = linear(x, 1, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            return x

    ## "For PN, the embedding architecture consists of two MLP layers with ReLU as the activation function." (Y-X Wang, 2018)
    def proto_encoder(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('pro', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
        return x
    
    def encoder_pose(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('hal_enc_pose', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
            return x
    
    ## Add a transformer to encode the pose_ref into seed class
    def transformer(self, input_pose, code_class, bn_train, with_BN=True, epsilon=1e-5, reuse=False):
        code_pose = self.encoder_pose(input_pose, bn_train=bn_train, with_BN=with_BN, reuse=reuse)
        x = tf.concat([code_class, code_pose], axis=1)
        with tf.variable_scope('hal_tran', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
            return x, code_pose
    
    def train(self,
              num_epoch=100,
              n_ite_per_epoch=600,
              lr_start=1e-5,
              lr_decay=0.5,
              lr_decay_step=20,
              patience=10):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'samples'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        # op = self.sess.graph.get_operations()
        # for m in op:
        #     print(m.values())
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### episodic training
        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []
        start_time = time.time()
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        best_val_loss = None
        for epoch in range(1, (num_epoch+1)):
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            loss_ite_val = []
            acc_ite_val = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                # if self.lambda_tf > 0:
                #     for i_d_update in range(self.d_per_g):
                #         _ = self.sess.run(self.opt_d,
                #                           feed_dict={self.bn_train_hal: False,
                #                                      self.learning_rate: lr})
                ##### make episode
                skip_this_episode = False
                selected_lbs = np.random.choice(list(self.all_train_labels), self.n_way, replace=False)
                selected_lbs_pose = np.random.choice(list(self.all_train_labels - set(selected_lbs)), self.n_way, replace=False)
                try:
                    selected_indexes = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs[lb_idx]], self.n_intra+self.n_shot+self.n_query_all//self.n_way, replace=False)) \
                                        for lb_idx in range(self.n_way)]
                    selected_indexes_pose = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs_pose[lb_idx]], self.n_aug-self.n_shot, replace=False)) \
                                             for lb_idx in range(self.n_way)]
                except:
                    print('[Training] Skip this episode since there are not enough samples for some label')
                    skip_this_episode = True
                if skip_this_episode:
                    continue
                pose_ref_intra_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][0:self.n_intra]] for lb_idx in range(self.n_way)])
                pose_ref_intra_features = np.reshape(pose_ref_intra_features, (self.n_way, self.n_intra, self.fc_dim))
                support_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][self.n_intra:(self.n_intra+self.n_shot)]] for lb_idx in range(self.n_way)])
                support_features = np.reshape(support_features, (self.n_way, self.n_shot, self.fc_dim))
                query_features = np.concatenate([self.train_feat_list[selected_indexes[lb_idx][self.n_intra+self.n_shot:]] for lb_idx in range(self.n_way)])
                query_labels = np.concatenate([np.repeat(lb_idx, self.n_query_all//self.n_way) for lb_idx in range(self.n_way)])
                pose_ref_features = np.concatenate([self.train_feat_list[selected_indexes_pose[lb_idx]] for lb_idx in range(self.n_way)])
                pose_ref_features = np.reshape(pose_ref_features, (self.n_way, self.n_aug-self.n_shot, self.fc_dim))
                support_labels = selected_lbs
                pose_ref_labels = selected_lbs_pose
                support_classes = np.array([self.train_class_list_raw[selected_indexes[i][0]] for i in range(self.n_way)])
                pose_ref_classes = np.array([self.train_class_list_raw[selected_indexes_pose[i][0]] for i in range(self.n_way)])
                _, loss, acc = self.sess.run([self.opt, self.loss_all, self.acc_pro_aug],
                                             feed_dict={self.support_features: support_features,
                                                        self.query_features: query_features,
                                                        self.query_labels: query_labels,
                                                        self.pose_ref_features: pose_ref_features,
                                                        self.pose_ref_intra_features: pose_ref_intra_features,
                                                        self.support_labels: support_labels,
                                                        self.pose_ref_labels: pose_ref_labels,
                                                        self.support_classes: support_classes,
                                                        self.pose_ref_classes: pose_ref_classes,
                                                        self.bn_train_hal: True,
                                                        self.learning_rate: lr})
                loss_ite_train.append(loss)
                acc_ite_train.append(np.mean(acc))
            
            ## validation on val classes
            # for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
            #     ##### make episode
            #     skip_this_episode = False
            #     selected_lbs = np.random.choice(list(self.all_val_labels), self.n_way, replace=False)
            #     selected_lbs_pose = np.random.choice(list(self.all_train_labels), self.n_way, replace=False)
            #     try:
            #         selected_indexes = [list(np.random.choice(self.candidate_indexes_each_lb_val[selected_lbs[lb_idx]], self.n_shot+self.n_query_all//self.n_way, replace=False)) \
            #                             for lb_idx in range(self.n_way)]
            #         selected_indexes_pose = [list(np.random.choice(self.candidate_indexes_each_lb_train[selected_lbs_pose[lb_idx]], self.n_aug-self.n_shot, replace=False)) \
            #                                  for lb_idx in range(self.n_way)]
            #     except:
            #         print('[Training] Skip this episode since there are not enough samples for some label')
            #         skip_this_episode = True
            #     if skip_this_episode:
            #         continue
            #     support_features = np.concatenate([self.val_feat_list[selected_indexes[lb_idx][0:self.n_shot]] for lb_idx in range(self.n_way)])
            #     support_features = np.reshape(support_features, (self.n_way, self.n_shot, self.fc_dim))
            #     query_features = np.concatenate([self.val_feat_list[selected_indexes[lb_idx][self.n_shot:]] for lb_idx in range(self.n_way)])
            #     query_labels = np.concatenate([np.repeat(lb_idx, self.n_query_all//self.n_way) for lb_idx in range(self.n_way)])
            #     pose_ref_features = np.concatenate([self.train_feat_list[selected_indexes_pose[lb_idx]] for lb_idx in range(self.n_way)])
            #     pose_ref_features = np.reshape(pose_ref_features, (self.n_way, self.n_aug-self.n_shot, self.fc_dim))
            #     loss, acc = self.sess.run([self.loss_pro_aug, self.acc_pro_aug],
            #                                  feed_dict={self.support_features: support_features,
            #                                             self.query_features: query_features,
            #                                             self.query_labels: query_labels,
            #                                             self.pose_ref_features: pose_ref_features,
            #                                             self.pose_ref_intra_features: pose_ref_intra_features,
            #                                             self.bn_train_hal: False})
            #     loss_ite_val.append(loss)
            #     acc_ite_val.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            # loss_val.append(np.mean(loss_ite_val))
            # acc_val.append(np.mean(acc_ite_val))
            # print('---- Epoch: %d, learning_rate: %f, training loss: %f, training accuracy: %f, validation loss: %f, validation accuracy: %f' % \
            #     (epoch, lr, loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))
            #### save model if improvement
            # if best_val_loss is None or loss_val[-1] < best_val_loss:
            #     best_val_loss = loss_val[-1]
            #     self.saver_hal_pro.save(self.sess,
            #                             os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
            #                             global_step=epoch)
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training accuracy: %f' % \
                (epoch, lr, loss_train[-1], acc_train[-1]))
        self.saver_hal_pro.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
                                global_step=epoch)
        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, acc_train]

class HAL_PN_PoseRef_Before(HAL_PN_PoseRef):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 val_path=None,
                 label_key='image_labels',
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 fc_dim=512,
                 l2scale=0.001,
                 n_train_class=64,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=4,
                 lambda_meta=1.0,
                 lambda_recon=1.0,
                 lambda_consistency=0.0,
                 lambda_consistency_pose=0.0,
                 lambda_intra=0.0,
                 lambda_pose_code_reg=0.0,
                 lambda_aux=0.0,
                 lambda_gan=0.0,
                 lambda_tf=0.0,
                 gp_scale=10.0,
                 d_per_g=5):
        super(HAL_PN_PoseRef_Before, self).__init__(sess,
                                                    model_name,
                                                    result_path,
                                                    train_path,
                                                    val_path,
                                                    label_key,
                                                    n_way,
                                                    n_shot,
                                                    n_aug,
                                                    n_query_all,
                                                    fc_dim,
                                                    l2scale,
                                                    n_train_class,
                                                    with_BN,
                                                    with_pro,
                                                    bnDecay,
                                                    epsilon,
                                                    num_parallel_calls,
                                                    lambda_meta,
                                                    lambda_recon,
                                                    lambda_consistency,
                                                    lambda_consistency_pose,
                                                    lambda_intra,
                                                    lambda_pose_code_reg,
                                                    lambda_aux,
                                                    lambda_gan,
                                                    lambda_tf,
                                                    gp_scale,
                                                    d_per_g)
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train_hal = tf.placeholder('bool', name='bn_train_hal')
        
        ### (1) episodic training
        ### dataset
        self.support_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_shot, self.fc_dim], name='support_features')
        self.query_features = tf.placeholder(tf.float32, shape=[self.n_query_all, self.fc_dim], name='query_features')
        self.query_labels = tf.placeholder(tf.int32, shape=[self.n_query_all], name='query_labels')
        self.pose_ref_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_aug-self.n_shot, self.fc_dim], name='pose_ref_features')
        self.pose_ref_intra_features = tf.placeholder(tf.float32, shape=[self.n_way, self.n_intra, self.fc_dim], name='pose_ref_intra_features')
        self.support_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='support_labels')
        self.pose_ref_labels = tf.placeholder(tf.int32, shape=[self.n_way], name='pose_ref_labels')
        self.support_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='support_classes')
        self.pose_ref_classes = tf.placeholder(tf.int32, shape=[self.n_way], name='pose_ref_classes')
        
        ### basic operation
        self.support_feat_flat = tf.reshape(self.support_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support_feat_ave = tf.reduce_mean(self.support_features, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.pose_ref_feat_flat = tf.reshape(self.pose_ref_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref_feat_ave = tf.reduce_mean(self.pose_ref_features, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.pose_ref_intra_feat_flat = tf.reshape(self.pose_ref_intra_features, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### hallucination flow
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_features, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
            self.pose_ref_class_code = self.proto_encoder(self.pose_ref_feat_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.support_class_code = self.support_feat_flat
            self.query_class_code = self.query_features
            self.pose_ref_class_code = self.pose_ref_feat_flat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_feat_ave_tile = tf.tile(self.support_feat_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_feat_ave_tile_flat = tf.reshape(self.support_feat_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.transformed_pose_ref, self.pose_code = self.transformer(self.pose_ref_feat_flat, self.support_feat_ave_tile_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.hallucinated_features = tf.reshape(self.transformed_pose_ref, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        ### prototypical network data flow using the augmented support set
        if self.with_pro:
            self.hal_class_code = self.proto_encoder(self.transformed_pose_ref, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.hal_class_code = self.transformed_pose_ref
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                                   logits=self.logits_pro_aug,
                                                                                   name='loss_pro_aug'))
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

        ### cycle-consistency loss
        self.pose_ref_feat_ave_tile = tf.tile(self.pose_ref_feat_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.pose_ref_feat_ave_tile_flat = tf.reshape(self.pose_ref_feat_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref_recon, self.transformed_pose_code = self.transformer(self.transformed_pose_ref, self.pose_ref_feat_ave_tile_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.loss_pose_ref_recon = tf.reduce_mean((self.pose_ref_feat_flat - self.pose_ref_recon)**2)
        self.loss_pose_code_recon = tf.reduce_mean((self.pose_code - self.transformed_pose_code)**2)
        ### reconstruction loss
        self.support_feat_ave_tile2 = tf.tile(self.support_feat_ave, multiples=[1, self.n_shot, 1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_feat_ave_tile2_flat = tf.reshape(self.support_feat_ave_tile2, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support_recon, _ = self.transformer(self.support_feat_flat, self.support_feat_ave_tile2_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True)
        self.loss_support_recon = tf.reduce_mean((self.support_feat_flat - self.support_recon)**2)
        ### intra-class transformation consistency loss
        ### (if pose-ref are of the same class as the support samples, then the transformed pose-ref should be as close to the pose-ref itself as possible)
        self.support_feat_ave_tile3 = tf.tile(self.support_feat_ave, multiples=[1, self.n_intra, 1]) ### shape: [self.n_way, self.n_intra, self.fc_dim]
        self.support_feat_ave_tile3_flat = tf.reshape(self.support_feat_ave_tile3, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_intra, self.fc_dim]
        self.transformed_pose_ref_intra, _ = self.transformer(self.pose_ref_intra_feat_flat, self.support_feat_ave_tile3_flat, bn_train=self.bn_train_hal, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*self.n_intra, self.fc_dim]
        self.loss_intra = tf.reduce_mean((self.pose_ref_intra_feat_flat - self.transformed_pose_ref_intra)**2)
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.loss_all = self.lambda_meta * self.loss_pro_aug + \
                        self.lambda_recon * self.loss_support_recon + \
                        self.lambda_consistency * self.loss_pose_ref_recon + \
                        self.lambda_consistency_pose * self.loss_pose_code_recon + \
                        self.lambda_intra * self.loss_intra
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_all+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
        
        ### model saver (keep the best checkpoint)
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)

        ### Count number of trainable variables
        total_params = 0
        for var in self.trainable_vars:
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total_params += var_params
        print('total number of parameters: %d' % total_params)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]

    ## Add a transformer to encode the pose_ref into seed class
    def transformer(self, input_pose, input_class, bn_train, with_BN=True, epsilon=1e-5, reuse=False):
        code_class = self.proto_encoder(input_class, bn_train=bn_train, with_BN=with_BN, reuse=True)
        code_pose = self.encoder_pose(input_pose, bn_train=bn_train, with_BN=with_BN, reuse=reuse)
        x = tf.concat([code_class, code_pose], axis=1)
        with tf.variable_scope('hal_tran', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
            return x, code_pose