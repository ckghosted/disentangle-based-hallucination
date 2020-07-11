import os, re, time, glob
import numpy as np
import tensorflow as tf
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

class HAL_PN_baseline(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 test_path,
                 label_key='image_labels',
                 bsize=128,
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_query_all=75, ## number of samples in the query set
                 n_way_t=5, ## number of classes in the support set for testing
                 n_shot_t=1, ## number of samples per class in the support set for testing
                 n_query_all_t=75, ## number of samples in the query set for testing
                 fc_dim=512,
                 img_size=84,
                 c_dim=3,
                 l2scale=0.001,
                 n_train_class=64,
                 n_test_class=16,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=8):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query_all = n_query_all
        self.fc_dim = fc_dim
        self.img_size = img_size
        self.c_dim = c_dim
        self.l2scale = l2scale
        self.n_train_class = n_train_class
        self.n_test_class = n_test_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.num_parallel_calls = num_parallel_calls
        
        ### (n_way, n_shot, n_query_all) for testing
        self.n_way_t = n_way_t
        self.n_shot_t = n_shot_t
        self.n_query_all_t = n_query_all_t
        
        ### prepare datasets
        self.train_path = train_path
        self.test_path = test_path
        self.label_key = label_key
        self.bsize = bsize
        ### Load all training-class data as training data
        with open(self.train_path, 'r') as reader:
            train_dict = json.loads(reader.read())
        self.train_image_list = train_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        self.train_class_list_raw = train_dict[self.label_key]
        all_train_class = sorted(set(self.train_class_list_raw))
        print('original train class labeling:')
        print(all_train_class)
        label_mapping = {}
        for new_lb in range(self.n_train_class):
            label_mapping[all_train_class[new_lb]] = new_lb
        self.train_label_list = np.array([label_mapping[old_lb] for old_lb in self.train_class_list_raw])
        print('new train class labeling:')
        print(sorted(set(self.train_label_list)))
        self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
        self.nRemainder = len(self.train_image_list) % self.bsize
        
        ### Load all testing-class data as testing data
        with open(self.test_path, 'r') as reader:
            test_dict = json.loads(reader.read())
        self.test_image_list = test_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_test_class-1}
        self.test_class_list_raw = test_dict[self.label_key]
        all_test_class = sorted(set(self.test_class_list_raw))
        print('original test class labeling:')
        print(all_test_class)
        label_mapping = {}
        for new_lb in range(self.n_test_class):
            label_mapping[all_test_class[new_lb]] = new_lb
        self.test_label_list = np.array([label_mapping[old_lb] for old_lb in self.test_class_list_raw])
        print('new test class labeling:')
        print(sorted(set(self.test_label_list)))
        self.nBatches_test = int(np.ceil(len(self.test_image_list) / self.bsize))
        self.nRemainder_test = len(self.test_image_list) % self.bsize

        ### [2020/03/21] make candidate indexes for each label
        self.candidate_indexes_each_lb_train = {}
        for lb in range(self.n_train_class):
            self.candidate_indexes_each_lb_train[lb] = [idx for idx in range(len(self.train_label_list)) if self.train_label_list[idx] == lb]
        self.candidate_indexes_each_lb_test = {}
        for lb in range(self.n_test_class):
            self.candidate_indexes_each_lb_test[lb] = [idx for idx in range(len(self.test_label_list)) if self.test_label_list[idx] == lb]
        self.all_train_labels = set(self.train_label_list)
        self.all_test_labels = set(self.test_label_list)
    
    def make_indices_for_each_episode(self, all_labels_set, candidate_indexes, n_way, n_shot, n_query_all):
        ###### sample n_way classes from the set of training classes
        selected_lbs = np.random.choice(list(all_labels_set), n_way, replace=False)
        try:
            selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_all//n_way, replace=False)) \
                                for lb_idx in range(n_way)]
            #### (optional) do not balance the number of query samples for all classes in each episode (need to write some codes to decide 'n_query_each_lb')
            # selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_each_lb[lb_idx], replace=False)) \
            #                     for lb_idx in range(n_way)]
            # ------------------------------------------------------------------------------------------------
            #### [2020/06/25] Follow IdeMe-Net (Chen CVPR 2019) to sample support and query indexes separately
            ####              (i.e., some query examples may appear in the support set)
            # selected_indexes_support = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot, replace=False)) \
            #                             for lb_idx in range(n_way)]
            # selected_indexes_query = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_query_all//n_way, replace=False)) \
            #                           for lb_idx in range(n_way)]
            # selected_indexes = np.concatenate((selected_indexes_support, selected_indexes_query), axis=1)
        except:
            print('[Training] Skip this episode since there are not enough samples for some label')
        return selected_indexes
    
    def train_episode_generator(self):
        while True:
            selected_indexes = self.make_indices_for_each_episode(self.all_train_labels,
                                                                  self.candidate_indexes_each_lb_train,
                                                                  self.n_way,
                                                                  self.n_shot,
                                                                  self.n_query_all)
            yield selected_indexes

    def test_episode_generator(self):
        while True:
            selected_indexes = self.make_indices_for_each_episode(self.all_test_labels,
                                                                  self.candidate_indexes_each_lb_test,
                                                                  self.n_way_t,
                                                                  self.n_shot_t,
                                                                  self.n_query_all_t)
            yield selected_indexes
    
    ## image loader for multiclass training
    def _parse_function(self, filename, label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        seed = np.random.randint(0, 2 ** 31 - 1)
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        ori_image_shape = tf.shape(img)
        img = tf.image.random_flip_left_right(img, seed=seed)
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.random_crop(img, ori_image_shape, seed=seed)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return img, label
    
    ## image loader for episodic training
    def _load_img(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.image.central_crop(img, 0.875)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return tf.expand_dims(tf.expand_dims(img, 0), 0) ### shape: [1, 1, 224, 224, 3]
    
    ## image de-transformation for visualization
    def _detransform(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (img*std+mean)*255
    
    def make_img_and_lb_from_train_idx(self, selected_indexes):
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_shot)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_shot, self.n_shot+self.n_query_all//self.n_way)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all//self.n_way])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        ### additional: support labels for loss_aux (labels AFTER re-numbering: {0, 1, ..., self.n_train_class-1})
        support_labels = [tf.gather(self.train_label_list, selected_indexes[i,0]) for i in range(self.n_way)]
        ### additional: absolute class labels for visualization (labels BEFORE re-numbering)
        support_classes = [tf.gather(self.train_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way)]
        return support_images, query_images, query_labels, support_labels, support_classes
    
    def make_img_and_lb_from_test_idx(self, selected_indexes):
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t, self.n_shot_t+self.n_query_all_t//self.n_way_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way_t)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all_t//self.n_way_t])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        ### additional: absolute class labels for visualization (labels BEFORE re-numbering)
        support_classes = [tf.gather(self.test_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way_t)]
        return support_images, query_images, query_labels, support_classes
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        
        ### model parameters
        image_dims = [self.img_size, self.img_size, self.c_dim]

        ### (1) pre-train the feature extraction network using training-class images
        ### dataset
        image_paths = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.train_label_list)
        dataset_pre = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset_pre = dataset_pre.map(self._parse_function,
                                      num_parallel_calls=self.num_parallel_calls).shuffle(len(self.train_image_list)).prefetch(self.bsize).repeat().batch(self.bsize)
        iterator_pre = dataset_pre.make_one_shot_iterator()
        self.pretrain_images, self.pretrain_labels = iterator_pre.get_next()
        ### operation
        self.pretrain_labels_vec = tf.one_hot(self.pretrain_labels, self.n_train_class)
        self.pretrain_feat = self.extractor(self.pretrain_images, bn_train=self.bn_train, with_BN=self.with_BN)
        self.logits_pre = self.linear_cls(self.pretrain_feat)
        self.loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pretrain_labels_vec,
                                                                               logits=self.logits_pre,
                                                                               name='loss_pre'))
        self.acc_pre = tf.nn.in_top_k(self.logits_pre, self.pretrain_labels, k=1)
        self.update_ops_pre = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ### (2) episodic training
        ### dataset
        dataset = tf.data.Dataset.from_generator(self.train_episode_generator, (tf.int32))
        dataset = dataset.map(self.make_img_and_lb_from_train_idx, num_parallel_calls=self.num_parallel_calls).batch(1)
        iterator = dataset.make_one_shot_iterator()
        support_images, query_images, query_labels, support_labels, support_classes = iterator.get_next()
        self.support_images = tf.squeeze(support_images, [0])
        self.query_images = tf.squeeze(query_images, [0])
        self.query_labels = tf.squeeze(query_labels, [0])
        self.support_labels = tf.squeeze(support_labels, [0])
        self.support_classes = tf.squeeze(support_classes, [0])
        ### basic operation
        self.support_images_reshape = tf.reshape(self.support_images, shape=[-1]+image_dims) ### shape: [self.n_way*self.n_shot, self.img_size, self.img_size, self.c_dim]
        self.support_feat = self.extractor(self.support_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support = tf.reshape(self.support_feat, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.query_feat = self.extractor(self.query_images, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_query_all, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat, bn_train=self.bn_train, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
        else:
            self.support_class_code = self.support_feat
            self.query_class_code = self.query_feat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_prototypes = tf.reduce_mean(self.support_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro = -tf.norm(self.support_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                               logits=self.logits_pro,
                                                                               name='loss_pro'))
        self.acc_pro = tf.nn.in_top_k(self.logits_pro, self.query_labels, k=1)
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if not op in self.update_ops_pre]
        
        ### (3) model parameters for testing
        ### dataset
        dataset_t = tf.data.Dataset.from_generator(self.test_episode_generator, (tf.int32))
        dataset_t = dataset_t.map(self.make_img_and_lb_from_test_idx, num_parallel_calls=self.num_parallel_calls)
        dataset_t = dataset_t.batch(1)
        iterator_t = dataset_t.make_one_shot_iterator()
        support_images_t, query_images_t, query_labels_t, support_classes_t = iterator_t.get_next()
        self.support_images_t = tf.squeeze(support_images_t, [0])
        self.query_images_t = tf.squeeze(query_images_t, [0])
        self.query_labels_t = tf.squeeze(query_labels_t, [0])
        self.support_classes_t = tf.squeeze(support_classes_t, [0])
        ### operation
        self.support_images_reshape_t = tf.reshape(self.support_images_t, shape=[-1]+image_dims)
        self.support_feat_t = self.extractor(self.support_images_reshape_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.support_t = tf.reshape(self.support_feat_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.query_feat_t = self.extractor(self.query_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.query_labels_vec_t = tf.one_hot(self.query_labels_t, self.n_way_t)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code_t = self.proto_encoder(self.support_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
            self.query_class_code_t = self.proto_encoder(self.query_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.support_class_code_t = self.support_feat_t
            self.query_class_code_t = self.query_feat_t
        self.support_encode_t = tf.reshape(self.support_class_code_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.support_prototypes_t = tf.reduce_mean(self.support_encode_t, axis=1)
        self.query_tile_t = tf.reshape(tf.tile(self.query_class_code_t, multiples=[1, self.n_way_t]), [self.n_query_all_t, self.n_way_t, -1])
        self.logits_pro_t = -tf.norm(self.support_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                 logits=self.logits_pro_t,
                                                                                 name='loss_pro_t'))
        self.acc_pro_t = tf.nn.in_top_k(self.logits_pro_t, self.query_labels_t, k=1)
        
        ### (4) data loader and operation for testing class set for visualization
        ### dataset
        image_paths_t = tf.convert_to_tensor(self.test_image_list, dtype=tf.string)
        dataset_pre_t = tf.data.Dataset.from_tensor_slices(image_paths_t)
        dataset_pre_t = dataset_pre_t.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_t = dataset_pre_t.make_one_shot_iterator()
        pretrain_images_t = iterator_pre_t.get_next()
        self.pretrain_images_t = tf.squeeze(pretrain_images_t, [1, 2])
        ### operation
        self.test_feat = self.extractor(self.pretrain_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.test_class_code = self.proto_encoder(self.test_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.test_class_code = self.test_feat

        ### (5) data loader and operation for training class for visualization
        ### dataset
        image_paths_train = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        dataset_pre_train = tf.data.Dataset.from_tensor_slices(image_paths_train)
        dataset_pre_train = dataset_pre_train.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_train = dataset_pre_train.make_one_shot_iterator()
        pretrain_images_train = iterator_pre_train.get_next()
        self.pretrain_images_train = tf.squeeze(pretrain_images_train, [1, 2])
        ### operation
        self.train_feat = self.extractor(self.pretrain_images_train, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.train_class_code = self.proto_encoder(self.train_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.train_class_code = self.train_feat

        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_ext = [var for var in self.all_vars if ('ext' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_ext = [var for var in self.trainable_vars if ('ext' in var.name or 'cls' in var.name)]
        self.trainable_vars_pro = [var for var in self.trainable_vars if ('ext' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_ext = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_pro = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'pro' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops_pre):
            self.opt_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pre+sum(self.used_regs_ext),
                                                                      var_list=self.trainable_vars_ext)
        with tf.control_dependencies(self.update_ops):
            self.opt_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pro+sum(self.used_regs_pro),
                                                                      var_list=self.trainable_vars_pro)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(var_list=self.all_vars, max_to_keep=1)
        self.saver_ext = tf.train.Saver(var_list=self.all_vars_ext, max_to_keep=1)

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
    
    ## ResNet-18
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('ext', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            #### x.shape = [-1, 84, 84, 3] or [-1, 32, 32, 3]
            #### conv1
            with tf.variable_scope('conv1'):
                x = conv2d(x, output_dim=64, k_h=7, k_w=7, d_h=2, d_w=2, add_bias=(~with_BN))
                #### x.shape = [-1, 42, 42, 64] or [-1, 16, 16, 64]
                if with_BN:
                    x = batch_norm(x, is_train=bn_train)
                x = tf.nn.relu(x)
                x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv2_x
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_2')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv3_x
            x = resblk_first(x, out_channel=128, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_2')
            #### x.shape = [-1, 11, 11, 128] or [-1, 4, 4, 128]
            #### conv4_x
            x = resblk_first(x, out_channel=256, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_2')
            #### x.shape = [-1, 6, 6, 256] or [-1, 2, 2, 256]
            #### conv5_x
            x = resblk_first(x, out_channel=512, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv5_2')
            #### x.shape = [-1, 3, 3, 512] or [-1, 1, 1, 512]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 512]
            return x
    
    ## linear classifier
    def linear_cls(self, x, reuse=False):
        with tf.variable_scope('cls', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.n_train_class, add_bias=True, name='dense1')
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
    
    def train(self,
              ext_from=None,
              ext_from_ckpt=None,
              num_epoch_pretrain=100,
              lr_start_pre=1e-3,
              lr_decay_pre=0.5,
              lr_decay_step_pre=10,
              num_epoch_noHal=0,
              num_epoch_hal=0,
              num_epoch_joint=100,
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
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ### load or pre-train the feature extractor
        if os.path.exists(ext_from):
            could_load_ext, checkpoint_counter_ext = self.load_ext(ext_from, ext_from_ckpt)
        else:
            print('ext_from: %s not exists, train feature extractor from scratch using training-class images' % ext_from)
            loss_pre = []
            acc_pre = []
            for epoch in range(1, (num_epoch_pretrain+1)):
                ##### Follow AFHN to use an initial learning rate 1e-3 which decays to the half every 10 epochs
                lr_pre = lr_start_pre * lr_decay_pre**((epoch-1)//lr_decay_step_pre)
                loss_pre_batch = []
                acc_pre_batch = []
                for idx in tqdm.tqdm(range(self.nBatches)):
                    _, loss, acc = self.sess.run([self.opt_pre, self.loss_pre, self.acc_pre],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr_pre})
                    loss_pre_batch.append(loss)
                    acc_pre_batch.append(np.mean(acc))
                ##### record training loss for each iteration (instead of each epoch)
                loss_pre.append(np.mean(loss_pre_batch))
                acc_pre.append(np.mean(acc_pre_batch))
                print('Epoch: %d (lr=%f), loss_pre: %f, acc_pre: %f' % (epoch, lr_pre, loss_pre[-1], acc_pre[-1]))
        
        ### episodic training
        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []
        best_test_loss = None
        start_time = time.time()
        num_epoch = num_epoch_noHal + num_epoch_hal + num_epoch_joint
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        for epoch in range(1, (num_epoch+1)):
            #### visualization
            if epoch % n_ep_per_visualization == 0:
                samples_filename = 'base_ep%03d' % epoch
                ##### extract all training-class features using the current feature extractor
                train_feat_all = []
                train_class_code_all = []
                for ite_feat_ext in range(self.nBatches):
                    train_feat, train_class_code = self.sess.run([self.train_feat, self.train_class_code],
                                                                 feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches - 1 and self.nRemainder > 0:
                        train_feat = train_feat[:self.nRemainder,:]
                        train_class_code = train_class_code[:self.nRemainder,:]
                    train_feat_all.append(train_feat)
                    train_class_code_all.append(train_class_code)
                train_feat_all = np.concatenate(train_feat_all, axis=0)
                train_class_code_all = np.concatenate(train_class_code_all, axis=0)
                ##### extract one episode
                # x_i_image, x_i, query_images, query_labels, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                x_i_image, x_i, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                                                                     self.support, ##### shape: [self.n_way, self.n_shot, self.fc_dim]
                                                                     # self.query_images,
                                                                     # self.query_labels,
                                                                     self.support_encode, ##### shape: [self.n_way, self.n_shot, self.fc_dim]
                                                                     self.support_classes], ##### shape: [self.n_way]
                                                                    feed_dict={self.bn_train: False})
                ##### real features t-SNE visualization
                seed_feat = {}
                for lb_idx in range(self.n_way):
                    abs_class = y_i[lb_idx]
                    seed_feat[abs_class] = x_i[lb_idx,:,:]
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_feat[lb], train_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_shot+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i[0:10],
                                 all_labels=sorted(set(self.train_class_list_raw)),
                                 n_shot=self.n_shot,
                                 n_min=self.n_shot,
                                 title='real features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_class_codes = {}
                    for lb_idx in range(self.n_way):
                        abs_class = y_i[lb_idx]
                        seed_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_class_codes[lb], train_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_shot+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i[0:10],
                                     all_labels=sorted(set(self.train_class_list_raw)),
                                     n_shot=self.n_shot,
                                     n_min=self.n_shot,
                                     title='real class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            #### training loops
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            if epoch <= num_epoch_noHal:
                ##### train the feature extractor and prototypical network without hallucination
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt_pro, self.loss_pro, self.acc_pro],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            elif epoch <= num_epoch_hal:
                print('no hallucinator for baseline model')
            else:
                print('no hallucinator for baseline model')
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            
            #### validation
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch,
                                                                          apply_hal=False,
                                                                          plot_samples=(epoch % n_ep_per_visualization == 0),
                                                                          samples_filename='valid_ep%03d'%epoch)
            loss_test.append(test_loss)
            acc_test.append(test_acc)
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training acc: %f (std: %f), valid loss: %f, valid accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (epoch, lr, loss_train[-1], acc_train[-1], np.std(acc_ite_train), test_loss, test_acc, test_acc_std, n_1_acc))
                
            #### save model if performance has improved
            if best_test_loss is None:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)
            elif test_loss < best_test_loss:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)

        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, loss_test, acc_train, acc_test]
    
    def run_testing(self, n_ite_per_epoch, apply_hal=False, plot_samples=False, samples_filename='sample'):
        loss_ite_test = []
        acc_ite_test = []
        n_1_acc = 0
        for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
            if plot_samples and ite == 1:
                ##### extract all testing-class features using the current feature extractor
                test_feat_all = []
                test_class_code_all = []
                for ite_feat_ext in range(self.nBatches_test):
                    test_feat, test_class_code = self.sess.run([self.test_feat, self.test_class_code],
                                                               feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches_test - 1 and self.nRemainder_test > 0:
                        test_feat = test_feat[:self.nRemainder_test,:]
                        test_class_code = test_class_code[:self.nRemainder_test,:]
                    test_feat_all.append(test_feat)
                    test_class_code_all.append(test_class_code)
                test_feat_all = np.concatenate(test_feat_all, axis=0)
                test_class_code_all = np.concatenate(test_class_code_all, axis=0)
                ##### run one testing eposide and extract support and query features and the corresponding absolute class labels
                # loss, acc, query_images_t, query_labels_t, x_i_image, x_i, y_i  = self.sess.run([self.loss_pro_t,
                loss, acc, x_i_image, x_i, x_class_codes, y_i  = self.sess.run([self.loss_pro_t,
                                                                                self.acc_pro_t,
                                                                                # self.query_images_t,
                                                                                # self.query_labels_t,
                                                                                self.support_images_t, ##### shape: [self.n_way_t, self.n_shot_t] + image_dims
                                                                                self.support_t, ##### shape: [self.n_way_t, self.n_shot_t, self.fc_dim]
                                                                                self.support_encode_t, ##### shape: [self.n_way_t, self.n_shot_t, self.fc_dim]
                                                                                self.support_classes_t], ##### shape: [self.n_way_t]
                                                                               feed_dict={self.bn_train: False})
                ##### real features t-SNE visualization
                seed_feat = {}
                for lb_idx in range(self.n_way_t):
                    abs_class = y_i[lb_idx]
                    seed_feat[abs_class] = x_i[lb_idx,:,:]
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_feat[lb], test_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_shot_t+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i,
                                 all_labels=sorted(set(self.test_class_list_raw)),
                                 n_shot=self.n_shot_t,
                                 n_min=self.n_shot_t,
                                 title='real features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))
                
                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_class_codes = {}
                    for lb_idx in range(self.n_way_t):
                        abs_class = y_i[lb_idx]
                        seed_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_class_codes[lb], test_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_shot_t+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i,
                                     all_labels=sorted(set(self.test_class_list_raw)),
                                     n_shot=self.n_shot_t,
                                     n_min=self.n_shot_t,
                                     title='real class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            else:
                ##### run one testing eposide without visualization
                loss, acc = self.sess.run([self.loss_pro_t, self.acc_pro_t],
                                          feed_dict={self.bn_train: False})
            loss_ite_test.append(loss)
            acc_ite_test.append(np.mean(acc))
            if np.mean(acc) > 0.99:
                n_1_acc += 1
        return np.mean(loss_ite_test), np.mean(acc_ite_test), np.std(acc_ite_test), n_1_acc
    
    def inference(self,
                  hal_from=None, ## e.g., model_name (must given)
                  hal_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  label_key='image_labels',
                  n_ite_per_epoch=600):
        ### load previous trained hallucinator
        could_load, checkpoint_counter = self.load(hal_from, hal_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch, apply_hal=False, plot_samples=True, samples_filename='novel')
            print('---- Novel (without hal) ---- novel loss: %f, novel accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (test_loss, test_acc, test_acc_std, n_1_acc))

    ## for loading the trained hallucinator and prototypical network
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    ## for loading the pre-trained feature extractor
    def load_ext(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_ext.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot(self, samples, n_row, n_col):
        fig = plt.figure(figsize=(n_col*2, n_row*2))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.img_size, self.img_size, 3))
        return fig

class HAL_PN_GAN(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 test_path,
                 label_key='image_labels',
                 bsize=128,
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 n_way_t=5, ## number of classes in the support set for testing
                 n_shot_t=1, ## number of samples per class in the support set for testing
                 n_aug_t=5, ## number of samples per class in the augmented support set for testing
                 n_query_all_t=75, ## number of samples in the query set for testing
                 fc_dim=512,
                 z_dim=512,
                 z_std=1.0,
                 img_size=84,
                 c_dim=3,
                 l2scale=0.001,
                 n_train_class=64,
                 n_test_class=16,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=8):
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
        self.z_dim = z_dim
        self.z_std = z_std
        self.img_size = img_size
        self.c_dim = c_dim
        self.l2scale = l2scale
        self.n_train_class = n_train_class
        self.n_test_class = n_test_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.num_parallel_calls = num_parallel_calls
        
        ### (n_way, n_shot, n_aug, n_query_all) for testing
        self.n_way_t = n_way_t
        self.n_shot_t = n_shot_t
        self.n_aug_t = n_aug_t
        self.n_query_all_t = n_query_all_t
        
        ### prepare datasets
        self.train_path = train_path
        self.test_path = test_path
        self.label_key = label_key
        self.bsize = bsize
        ### Load all training-class data as training data
        with open(self.train_path, 'r') as reader:
            train_dict = json.loads(reader.read())
        self.train_image_list = train_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        self.train_class_list_raw = train_dict[self.label_key]
        all_train_class = sorted(set(self.train_class_list_raw))
        print('original train class labeling:')
        print(all_train_class)
        label_mapping = {}
        for new_lb in range(self.n_train_class):
            label_mapping[all_train_class[new_lb]] = new_lb
        self.train_label_list = np.array([label_mapping[old_lb] for old_lb in self.train_class_list_raw])
        print('new train class labeling:')
        print(sorted(set(self.train_label_list)))
        self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
        self.nRemainder = len(self.train_image_list) % self.bsize
        
        ### Load all testing-class data as testing data
        with open(self.test_path, 'r') as reader:
            test_dict = json.loads(reader.read())
        self.test_image_list = test_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_test_class-1}
        self.test_class_list_raw = test_dict[self.label_key]
        all_test_class = sorted(set(self.test_class_list_raw))
        print('original test class labeling:')
        print(all_test_class)
        label_mapping = {}
        for new_lb in range(self.n_test_class):
            label_mapping[all_test_class[new_lb]] = new_lb
        self.test_label_list = np.array([label_mapping[old_lb] for old_lb in self.test_class_list_raw])
        print('new test class labeling:')
        print(sorted(set(self.test_label_list)))
        self.nBatches_test = int(np.ceil(len(self.test_image_list) / self.bsize))
        self.nRemainder_test = len(self.test_image_list) % self.bsize

        ### [2020/03/21] make candidate indexes for each label
        self.candidate_indexes_each_lb_train = {}
        for lb in range(self.n_train_class):
            self.candidate_indexes_each_lb_train[lb] = [idx for idx in range(len(self.train_label_list)) if self.train_label_list[idx] == lb]
        self.candidate_indexes_each_lb_test = {}
        for lb in range(self.n_test_class):
            self.candidate_indexes_each_lb_test[lb] = [idx for idx in range(len(self.test_label_list)) if self.test_label_list[idx] == lb]
        self.all_train_labels = set(self.train_label_list)
        self.all_test_labels = set(self.test_label_list)
    
    def make_indices_for_train_episode(self, all_labels_set, candidate_indexes, n_way, n_shot, n_query_all):
        ###### sample n_way classes from the set of training classes
        selected_lbs = np.random.choice(list(all_labels_set), n_way, replace=False)
        try:
            selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_all//n_way, replace=False)) \
                                for lb_idx in range(n_way)]
            #### (optional) do not balance the number of query samples for all classes in each episode
            # selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_each_lb[lb_idx], replace=False)) \
            #                     for lb_idx in range(n_way)]
        except:
            print('[Training] Skip this episode since there are not enough samples for some label')
        return selected_indexes

    def make_indices_for_test_episode(self, all_labels_set, candidate_indexes, n_way, n_shot, n_query_all):
        ###### sample n_way classes from the set of testing classes
        selected_lbs = np.random.choice(list(all_labels_set), n_way, replace=False)
        try:
            selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_all//n_way, replace=False)) \
                                for lb_idx in range(n_way)]
            #### (optional) do not balance the number of query samples for all classes in each episode
            # selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_each_lb[lb_idx], replace=False)) \
            #                     for lb_idx in range(n_way)]
        except:
            print('[Training] Skip this episode since there are not enough samples for some label')
        return selected_indexes
    
    def train_episode_generator(self):
        while True:
            selected_indexes = self.make_indices_for_train_episode(self.all_train_labels,
                                                                   self.candidate_indexes_each_lb_train,
                                                                   self.n_way,
                                                                   self.n_shot,
                                                                   self.n_query_all)
            yield selected_indexes

    def test_episode_generator(self):
        while True:
            selected_indexes = self.make_indices_for_test_episode(self.all_test_labels,
                                                                  self.candidate_indexes_each_lb_test,
                                                                  self.n_way_t,
                                                                  self.n_shot_t,
                                                                  self.n_query_all_t)
            yield selected_indexes
    
    ## image loader for multiclass training
    def _parse_function(self, filename, label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        seed = np.random.randint(0, 2 ** 31 - 1)
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        ori_image_shape = tf.shape(img)
        img = tf.image.random_flip_left_right(img, seed=seed)
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.random_crop(img, ori_image_shape, seed=seed)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return img, label
    
    ## image loader for episodic training
    def _load_img(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.image.central_crop(img, 0.875)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return tf.expand_dims(tf.expand_dims(img, 0), 0) ### shape: [1, 1, 224, 224, 3]

    def _detransform(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (img*std+mean)*255
    
    def make_img_and_lb_from_train_idx(self, selected_indexes):
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_shot)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_shot, self.n_shot+self.n_query_all//self.n_way)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all//self.n_way])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        ### additional: support labels for loss_aux (labels AFTER re-numbering: {0, 1, ..., self.n_train_class-1})
        support_labels = [tf.gather(self.train_label_list, selected_indexes[i,0]) for i in range(self.n_way)]
        ### additional: absolute class labels for visualization
        support_classes = [tf.gather(self.train_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way)]
        return support_images, query_images, query_labels, support_labels, support_classes
    
    def make_img_and_lb_from_test_idx(self, selected_indexes):
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t, self.n_shot_t+self.n_query_all_t//self.n_way_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way_t)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all_t//self.n_way_t])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        ### additional: absolute class labels for visualization
        support_classes = [tf.gather(self.test_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way_t)]
        return support_images, query_images, query_labels, support_classes
    
    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        
        ### model parameters
        image_dims = [self.img_size, self.img_size, self.c_dim]

        ### (1) pre-train the feature extraction network using training-class images
        ### dataset
        image_paths = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.train_label_list)
        dataset_pre = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset_pre = dataset_pre.map(self._parse_function,
                                      num_parallel_calls=self.num_parallel_calls).shuffle(len(self.train_image_list)).prefetch(self.bsize).repeat().batch(self.bsize)
        iterator_pre = dataset_pre.make_one_shot_iterator()
        self.pretrain_images, self.pretrain_labels = iterator_pre.get_next()
        ### operation
        self.pretrain_labels_vec = tf.one_hot(self.pretrain_labels, self.n_train_class)
        self.pretrain_feat = self.extractor(self.pretrain_images, bn_train=self.bn_train, with_BN=self.with_BN)
        self.logits_pre = self.linear_cls(self.pretrain_feat)
        self.loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pretrain_labels_vec,
                                                                               logits=self.logits_pre,
                                                                               name='loss_pre'))
        self.acc_pre = tf.nn.in_top_k(self.logits_pre, self.pretrain_labels, k=1)
        self.update_ops_pre = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ### (2) episodic training
        ### dataset
        dataset = tf.data.Dataset.from_generator(self.train_episode_generator, (tf.int32))
        dataset = dataset.map(self.make_img_and_lb_from_train_idx, num_parallel_calls=self.num_parallel_calls).batch(1)
        iterator = dataset.make_one_shot_iterator()
        support_images, query_images, query_labels, support_labels, support_classes = iterator.get_next()
        self.support_images = tf.squeeze(support_images, [0])
        self.query_images = tf.squeeze(query_images, [0])
        self.query_labels = tf.squeeze(query_labels, [0])
        self.support_labels = tf.squeeze(support_labels, [0])
        self.support_classes = tf.squeeze(support_classes, [0])
        ### basic operation
        self.support_images_reshape = tf.reshape(self.support_images, shape=[-1]+image_dims) ### shape: [self.n_way*self.n_shot, self.img_size, self.img_size, self.c_dim]
        self.support_feat = self.extractor(self.support_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support = tf.reshape(self.support_feat, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.query_feat = self.extractor(self.query_images, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_query_all, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat, bn_train=self.bn_train, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
        else:
            self.support_class_code = self.support_feat
            self.query_class_code = self.query_feat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_prototypes = tf.reduce_mean(self.support_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro = -tf.norm(self.support_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                               logits=self.logits_pro,
                                                                               name='loss_pro'))
        self.acc_pro = tf.nn.in_top_k(self.logits_pro, self.query_labels, k=1)
        ### hallucination flow
        self.hal_feat = self.build_augmentor(self.support, self.n_way, self.n_shot, self.n_aug) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.hallucinated_features = tf.reshape(self.hal_feat, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        ### prototypical network data flow using the augmented support set
        if self.with_pro:
            self.hal_class_code = self.proto_encoder(self.hal_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.hal_class_code = self.hal_feat
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                                   logits=self.logits_pro_aug,
                                                                                   name='loss_pro_aug'))
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if not op in self.update_ops_pre]
        
        ### (3) model parameters for testing
        ### dataset
        dataset_t = tf.data.Dataset.from_generator(self.test_episode_generator, (tf.int32))
        dataset_t = dataset_t.map(self.make_img_and_lb_from_test_idx, num_parallel_calls=self.num_parallel_calls)
        dataset_t = dataset_t.batch(1)
        iterator_t = dataset_t.make_one_shot_iterator()
        support_images_t, query_images_t, query_labels_t, support_classes_t = iterator_t.get_next()
        self.support_images_t = tf.squeeze(support_images_t, [0])
        self.query_images_t = tf.squeeze(query_images_t, [0])
        self.query_labels_t = tf.squeeze(query_labels_t, [0])
        self.support_classes_t = tf.squeeze(support_classes_t, [0])
        ### operation
        self.support_images_reshape_t = tf.reshape(self.support_images_t, shape=[-1]+image_dims)
        self.support_feat_t = self.extractor(self.support_images_reshape_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.support_t = tf.reshape(self.support_feat_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.query_feat_t = self.extractor(self.query_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.query_labels_vec_t = tf.one_hot(self.query_labels_t, self.n_way_t)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code_t = self.proto_encoder(self.support_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
            self.query_class_code_t = self.proto_encoder(self.query_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.support_class_code_t = self.support_feat_t
            self.query_class_code_t = self.query_feat_t
        self.support_encode_t = tf.reshape(self.support_class_code_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.support_prototypes_t = tf.reduce_mean(self.support_encode_t, axis=1)
        self.query_tile_t = tf.reshape(tf.tile(self.query_class_code_t, multiples=[1, self.n_way_t]), [self.n_query_all_t, self.n_way_t, -1])
        self.logits_pro_t = -tf.norm(self.support_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                 logits=self.logits_pro_t,
                                                                                 name='loss_pro_t'))
        self.acc_pro_t = tf.nn.in_top_k(self.logits_pro_t, self.query_labels_t, k=1)
        self.hal_feat_t = self.build_augmentor(self.support_t, self.n_way_t, self.n_shot_t, self.n_aug_t, reuse=True)
        self.hallucinated_features_t = tf.reshape(self.hal_feat_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])
        if self.with_pro:
            self.hal_class_code_t = self.proto_encoder(self.hal_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.hal_class_code_t = self.hal_feat_t
        self.hal_encode_t = tf.reshape(self.hal_class_code_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])
        self.support_aug_encode_t = tf.concat((self.support_encode_t, self.hal_encode_t), axis=1)
        self.support_aug_prototypes_t = tf.reduce_mean(self.support_aug_encode_t, axis=1)
        self.logits_pro_aug_t = -tf.norm(self.support_aug_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_aug_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                     logits=self.logits_pro_aug_t,
                                                                                     name='loss_pro_aug_t'))
        self.acc_pro_aug_t = tf.nn.in_top_k(self.logits_pro_aug_t, self.query_labels_t, k=1)

        
        ### (4) data loader and operation for testing class set for visualization
        ### dataset
        image_paths_t = tf.convert_to_tensor(self.test_image_list, dtype=tf.string)
        dataset_pre_t = tf.data.Dataset.from_tensor_slices(image_paths_t)
        dataset_pre_t = dataset_pre_t.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_t = dataset_pre_t.make_one_shot_iterator()
        pretrain_images_t = iterator_pre_t.get_next()
        self.pretrain_images_t = tf.squeeze(pretrain_images_t, [1, 2])
        ### operation
        self.test_feat = self.extractor(self.pretrain_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.test_class_code = self.proto_encoder(self.test_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.test_class_code = self.test_feat

        ### (5) data loader and operation for training class for visualization
        ### dataset
        image_paths_train = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        dataset_pre_train = tf.data.Dataset.from_tensor_slices(image_paths_train)
        dataset_pre_train = dataset_pre_train.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_train = dataset_pre_train.make_one_shot_iterator()
        pretrain_images_train = iterator_pre_train.get_next()
        self.pretrain_images_train = tf.squeeze(pretrain_images_train, [1, 2])
        ### operation
        self.train_feat = self.extractor(self.pretrain_images_train, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.train_class_code = self.proto_encoder(self.train_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.train_class_code = self.train_feat
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_ext = [var for var in self.all_vars if ('ext' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_ext = [var for var in self.trainable_vars if ('ext' in var.name or 'cls' in var.name)]
        self.trainable_vars_hal = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name or 'cls' in var.name)]
        self.trainable_vars_pro = [var for var in self.trainable_vars if ('ext' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_ext = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_pro = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'pro' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops_pre):
            self.opt_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pre+sum(self.used_regs_ext),
                                                                      var_list=self.trainable_vars_ext)
        with tf.control_dependencies(self.update_ops):
            self.opt_hal = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pro_aug+sum(self.used_regs_hal),
                                                                      var_list=self.trainable_vars_hal)
            self.opt_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pro+sum(self.used_regs_pro),
                                                                      var_list=self.trainable_vars_pro)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_pro_aug+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(var_list=self.all_vars, max_to_keep=1)
        self.saver_ext = tf.train.Saver(var_list=self.all_vars_ext, max_to_keep=1)

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
    
    ## ResNet-18
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('ext', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            #### x.shape = [-1, 84, 84, 3] or [-1, 32, 32, 3]
            #### conv1
            with tf.variable_scope('conv1'):
                x = conv2d(x, output_dim=64, k_h=7, k_w=7, d_h=2, d_w=2, add_bias=(~with_BN))
                #### x.shape = [-1, 42, 42, 64] or [-1, 16, 16, 64]
                if with_BN:
                    x = batch_norm(x, is_train=bn_train)
                x = tf.nn.relu(x)
                x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv2_x
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_2')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv3_x
            x = resblk_first(x, out_channel=128, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_2')
            #### x.shape = [-1, 11, 11, 128] or [-1, 4, 4, 128]
            #### conv4_x
            x = resblk_first(x, out_channel=256, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_2')
            #### x.shape = [-1, 6, 6, 256] or [-1, 2, 2, 256]
            #### conv5_x
            x = resblk_first(x, out_channel=512, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv5_2')
            #### x.shape = [-1, 3, 3, 512] or [-1, 1, 1, 512]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 512]
            return x
    
    ## linear classifier
    def linear_cls(self, x, reuse=False):
        with tf.variable_scope('cls', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.n_train_class, add_bias=True, name='dense1')
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
        sampled_support = tf.reshape(sampled_support, shape=[n_way*(n_aug-n_shot), -1]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([sampled_support, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat
    
    def train(self,
              ext_from=None,
              ext_from_ckpt=None,
              num_epoch_pretrain=100,
              lr_start_pre=1e-3,
              lr_decay_pre=0.5,
              lr_decay_step_pre=10,
              num_epoch_noHal=0,
              num_epoch_hal=0,
              num_epoch_joint=100,
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
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ### load or pre-train the feature extractor
        if os.path.exists(ext_from):
            could_load_ext, checkpoint_counter_ext = self.load_ext(ext_from, ext_from_ckpt)
        else:
            print('ext_from: %s not exists, train feature extractor from scratch using training-class images' % ext_from)
            loss_pre = []
            acc_pre = []
            for epoch in range(1, (num_epoch_pretrain+1)):
                ##### Follow AFHN to use an initial learning rate 1e-3 which decays to the half every 10 epochs
                lr_pre = lr_start_pre * lr_decay_pre**((epoch-1)//lr_decay_step_pre)
                loss_pre_batch = []
                acc_pre_batch = []
                for idx in tqdm.tqdm(range(self.nBatches)):
                    _, loss, acc = self.sess.run([self.opt_pre, self.loss_pre, self.acc_pre],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr_pre})
                    loss_pre_batch.append(loss)
                    acc_pre_batch.append(np.mean(acc))
                ##### record training loss for each iteration (instead of each epoch)
                loss_pre.append(np.mean(loss_pre_batch))
                acc_pre.append(np.mean(acc_pre_batch))
                print('Epoch: %d (lr=%f), loss_pre: %f, acc_pre: %f' % (epoch, lr_pre, loss_pre[-1], acc_pre[-1]))
        
        ### episodic training
        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []
        best_test_loss = None
        start_time = time.time()
        num_epoch = num_epoch_noHal + num_epoch_hal + num_epoch_joint
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        for epoch in range(1, (num_epoch+1)):
            #### visualization
            if epoch % n_ep_per_visualization == 0:
                samples_filename = 'base_ep%03d' % epoch
                ##### extract all training-class features using the current feature extractor
                train_feat_all = []
                train_class_code_all = []
                for ite_feat_ext in range(self.nBatches):
                    train_feat, train_class_code = self.sess.run([self.train_feat, self.train_class_code],
                                                                 feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches - 1 and self.nRemainder > 0:
                        train_feat = train_feat[:self.nRemainder,:]
                        train_class_code = train_class_code[:self.nRemainder,:]
                    train_feat_all.append(train_feat)
                    train_class_code_all.append(train_class_code)
                train_feat_all = np.concatenate(train_feat_all, axis=0)
                train_class_code_all = np.concatenate(train_class_code_all, axis=0)
                ##### extract one episode
                # x_i_image, x_i, query_images, query_labels, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                x_i_image, x_i, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                                                                                self.support, ##### shape: [self.n_way, self.n_shot, self.fc_dim]
                                                                                # self.query_images,
                                                                                # self.query_labels,
                                                                                self.hallucinated_features, ##### shape: [self.n_way, self.n_aug - self.n_shot, self.fc_dim]
                                                                                self.support_aug_encode, ##### shape: [self.n_way, self.n_aug, self.fc_dim]
                                                                                self.support_classes], ##### shape: [self.n_way]
                                                                               feed_dict={self.bn_train: False})                
                ##### (x_i_image, x_j_image, nearest image to x_tilde_i) visualization
                x_dim = 224
                x_tilde_i_class_codes = x_class_codes[:,self.n_shot:,:]
                n_class_plot = self.n_way if self.n_way <= 10 else 10
                img_array = np.empty((3*n_class_plot, x_dim, x_dim, 3), dtype='uint8')
                nearest_class_list = []
                nearest_class_list_using_class_code = []
                for lb_idx in range(n_class_plot):
                    # x_i_image
                    img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                    # nearest image to x_tilde_i
                    nearest_idx = (np.sum(np.abs(train_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+n_class_plot,:] = img
                    nearest_class_list.append(str(self.train_class_list_raw[nearest_idx]))
                    # nearest image to x_tilde_i_class_codes
                    nearest_idx = (np.sum(np.abs(train_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+2*n_class_plot,:] = img
                    nearest_class_list_using_class_code.append(str(self.train_class_list_raw[nearest_idx]))
                subtitle_list = [str(y) for y in y_i[0:n_class_plot]] + nearest_class_list + nearest_class_list_using_class_code
                fig = plot(img_array, 3, n_class_plot, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
                
                ##### real and hallucinated features t-SNE visualization
                seed_and_hal_feat = {}
                for lb_idx in range(self.n_way):
                    abs_class = y_i[lb_idx]
                    seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], train_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i[0:10],
                                 all_labels=sorted(set(self.train_class_list_raw)),
                                 n_shot=self.n_shot,
                                 n_min=self.n_aug,
                                 title='real and hallucinated features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_and_hal_class_codes = {}
                    for lb_idx in range(self.n_way):
                        abs_class = y_i[lb_idx]
                        seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], train_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i[0:10],
                                     all_labels=sorted(set(self.train_class_list_raw)),
                                     n_shot=self.n_shot,
                                     n_min=self.n_aug,
                                     title='real and hallucinated class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            #### training loops
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            if epoch <= num_epoch_noHal:
                ##### train the feature extractor and prototypical network without hallucination
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt_pro, self.loss_pro, self.acc_pro],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            elif epoch <= num_epoch_hal:
                ##### freeze feature extractor and train hallucinator only
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt_hal, self.loss_pro_aug, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            else:
                ##### Train the whole model
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt, self.loss_pro_aug, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            
            #### validation
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch,
                                                                          apply_hal=(epoch > num_epoch_noHal),
                                                                          plot_samples=(epoch % n_ep_per_visualization == 0),
                                                                          samples_filename='valid_ep%03d'%epoch)
            loss_test.append(test_loss)
            acc_test.append(test_acc)
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training acc: %f (std: %f), valid loss: %f, valid accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (epoch, lr, loss_train[-1], acc_train[-1], np.std(acc_ite_train), test_loss, test_acc, test_acc_std, n_1_acc))
                
            #### save model if performance has improved
            if best_test_loss is None:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)
            elif test_loss < best_test_loss:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)

        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, loss_test, acc_train, acc_test]
    
    def run_testing(self, n_ite_per_epoch, apply_hal=False, plot_samples=False, samples_filename='sample'):
        loss_ite_test = []
        acc_ite_test = []
        n_1_acc = 0
        for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
            if apply_hal and plot_samples and ite == 1:
                ##### extract all testing-class features using the current feature extractor
                test_feat_all = []
                test_class_code_all = []
                for ite_feat_ext in range(self.nBatches_test):
                    test_feat, test_class_code = self.sess.run([self.test_feat, self.test_class_code],
                                                             feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches_test - 1 and self.nRemainder_test > 0:
                        test_feat = test_feat[:self.nRemainder_test,:]
                        test_class_code = test_class_code[:self.nRemainder_test,:]
                    test_feat_all.append(test_feat)
                    test_class_code_all.append(test_class_code)
                test_feat_all = np.concatenate(test_feat_all, axis=0)
                test_class_code_all = np.concatenate(test_class_code_all, axis=0)
                ##### run one testing eposide and extract hallucinated features and the corresponding absolute class labels
                # loss, acc, x_i_image, x_i, query_images_t, query_labels_t, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.loss_pro_aug_t,
                loss, acc, x_i_image, x_i, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.loss_pro_aug_t,
                                                                                           self.acc_pro_aug_t,
                                                                                           self.support_images_t, ##### shape: [self.n_way_t, self.n_shot_t] + image_dims
                                                                                           self.support_t, ##### shape: [self.n_way_t, self.n_shot_t, self.fc_dim]
                                                                                           # self.query_images_t,
                                                                                           # self.query_labels_t,
                                                                                           self.hallucinated_features_t, ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t, self.fc_dim]
                                                                                           self.support_aug_encode_t, ##### shape: [self.n_way_t, self.n_aug_t, self.fc_dim]
                                                                                           self.support_classes_t], ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t]
                                                                                          feed_dict={self.bn_train: False})
                ##### (x_i_image, x_j_image, nearest image to x_tilde_i) visualization
                x_dim = 224
                x_tilde_i_class_codes = x_class_codes[:,self.n_shot_t:,:]
                # img_array = np.empty((3*self.n_way_t, x_dim, x_dim, 3), dtype='uint8')
                # nearest_class_list = []
                # nearest_class_list_using_class_code = []
                # for lb_idx in range(self.n_way_t):
                #     # x_i_image
                #     img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                #     # nearest image to x_tilde_i
                #     nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
                #     file_path = self.test_image_list[nearest_idx]
                #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     img_array[lb_idx+self.n_way_t,:] = img
                #     nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
                #     # nearest image to x_tilde_i_class_codes
                #     nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
                #     file_path = self.test_image_list[nearest_idx]
                #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     img_array[lb_idx+2*self.n_way_t,:] = img
                #     nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
                # subtitle_list = [str(y) for y in y_i] + nearest_class_list + nearest_class_list_using_class_code
                # fig = plot(img_array, 3, self.n_way_t, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                # plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
                n_hal_per_class = self.n_aug_t - self.n_shot_t if self.n_aug_t - self.n_shot_t < 5 else 5
                for lb_idx in range(len(y_i[0:10])):
                    img_array = np.empty((3*n_hal_per_class, x_dim, x_dim, 3), dtype='uint8')
                    nearest_class_list = []
                    nearest_class_list_using_class_code = []
                    for aug_idx in range(n_hal_per_class):
                        img_array[aug_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                        nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,aug_idx,:]), axis=1)).argmin()
                        file_path = self.test_image_list[nearest_idx]
                        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_array[aug_idx+n_hal_per_class,:] = img
                        nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
                        nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,aug_idx,:]), axis=1)).argmin()
                        file_path = self.test_image_list[nearest_idx]
                        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_array[aug_idx+2*n_hal_per_class,:] = img
                        nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
                    subtitle_list = [str(y_i[lb_idx]) for _ in range(n_hal_per_class)] + nearest_class_list + nearest_class_list_using_class_code
                    fig = plot(img_array, 3, n_hal_per_class, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                    plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+('_%03d' % y_i[lb_idx])+'.png'), bbox_inches='tight')
                
                ##### real and hallucinated features t-SNE visualization
                seed_and_hal_feat = {}
                for lb_idx in range(self.n_way_t):
                    abs_class = y_i[lb_idx]
                    seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], test_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i,
                                 all_labels=sorted(set(self.test_class_list_raw)),
                                 n_shot=self.n_shot_t,
                                 n_min=self.n_aug_t,
                                 title='real and hallucinated features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_and_hal_class_codes = {}
                    for lb_idx in range(self.n_way_t):
                        abs_class = y_i[lb_idx]
                        seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], test_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i,
                                     all_labels=sorted(set(self.test_class_list_raw)),
                                     n_shot=self.n_shot_t,
                                     n_min=self.n_aug_t,
                                     title='real and hallucinated class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            elif apply_hal:
                ##### run one testing eposide with haluucination
                loss, acc = self.sess.run([self.loss_pro_aug_t, self.acc_pro_aug_t],
                                          feed_dict={self.bn_train: False})
            else:
                ##### run one testing eposide without haluucination
                loss, acc = self.sess.run([self.loss_pro_t, self.acc_pro_t],
                                          feed_dict={self.bn_train: False})
            loss_ite_test.append(loss)
            acc_ite_test.append(np.mean(acc))
            if np.mean(acc) > 0.99:
                n_1_acc += 1
        return np.mean(loss_ite_test), np.mean(acc_ite_test), np.std(acc_ite_test), n_1_acc
    
    def inference(self,
                  hal_from=None, ## e.g., model_name (must given)
                  hal_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  label_key='image_labels',
                  n_ite_per_epoch=600):
        ### load previous trained hallucinator
        could_load, checkpoint_counter = self.load(hal_from, hal_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch, apply_hal=False, plot_samples=True, samples_filename='novel')
            print('---- Novel (without hal) ---- novel loss: %f, novel accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (test_loss, test_acc, test_acc_std, n_1_acc))
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch, apply_hal=True, plot_samples=True, samples_filename='novel')
            print('---- Novel (with hal) ---- novel loss: %f, novel accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (test_loss, test_acc, test_acc_std, n_1_acc))

    ## for loading the trained hallucinator and prototypical network
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    ## for loading the pre-trained feature extractor
    def load_ext(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_ext.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot(self, samples, n_row, n_col):
        fig = plt.figure(figsize=(n_col*2, n_row*2))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.img_size, self.img_size, 3))
        return fig

class HAL_PN_GAN2(HAL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 test_path,
                 label_key='image_labels',
                 bsize=128,
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 n_way_t=5, ## number of classes in the support set for testing
                 n_shot_t=1, ## number of samples per class in the support set for testing
                 n_aug_t=5, ## number of samples per class in the augmented support set for testing
                 n_query_all_t=75, ## number of samples in the query set for testing
                 fc_dim=512,
                 z_dim=512,
                 z_std=0.1,
                 img_size=84,
                 c_dim=3,
                 l2scale=0.001,
                 n_train_class=64,
                 n_test_class=16,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=8):
        super(HAL_PN_GAN2, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          train_path,
                                          test_path,
                                          label_key,
                                          bsize,
                                          n_way,
                                          n_shot,
                                          n_aug,
                                          n_query_all,
                                          n_way_t,
                                          n_shot_t,
                                          n_aug_t,
                                          n_query_all_t,
                                          fc_dim,
                                          z_dim,
                                          z_std,
                                          img_size,
                                          c_dim,
                                          l2scale,
                                          n_train_class,
                                          n_test_class,
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
                 test_path,
                 label_key='image_labels',
                 bsize=128,
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 n_way_t=5, ## number of classes in the support set for testing
                 n_shot_t=1, ## number of samples per class in the support set for testing
                 n_aug_t=5, ## number of samples per class in the augmented support set for testing
                 n_query_all_t=75, ## number of samples in the query set for testing
                 fc_dim=512,
                 z_dim=512,
                 z_std=0.1,
                 img_size=84,
                 c_dim=3,
                 lambda_meta=1.0,
                 lambda_tf=0.0,
                 lambda_ar=0.0,
                 gp_scale=10.0,
                 d_per_g=5,
                 l2scale=0.001,
                 n_train_class=64,
                 n_test_class=16,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=8):
        super(HAL_PN_AFHN, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          train_path,
                                          test_path,
                                          label_key,
                                          bsize,
                                          n_way,
                                          n_shot,
                                          n_aug,
                                          n_query_all,
                                          n_way_t,
                                          n_shot_t,
                                          n_aug_t,
                                          n_query_all_t,
                                          fc_dim,
                                          z_dim,
                                          z_std,
                                          img_size,
                                          c_dim,
                                          l2scale,
                                          n_train_class,
                                          n_test_class,
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
        self.bn_train = tf.placeholder('bool', name='bn_train')
        
        ### model parameters
        image_dims = [self.img_size, self.img_size, self.c_dim]

        ### (1) pre-train the feature extraction network using training-class images
        ### dataset
        image_paths = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.train_label_list)
        dataset_pre = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset_pre = dataset_pre.map(self._parse_function,
                                      num_parallel_calls=self.num_parallel_calls).shuffle(len(self.train_image_list)).prefetch(self.bsize).repeat().batch(self.bsize)
        iterator_pre = dataset_pre.make_one_shot_iterator()
        self.pretrain_images, self.pretrain_labels = iterator_pre.get_next()
        ### operation
        self.pretrain_labels_vec = tf.one_hot(self.pretrain_labels, self.n_train_class)
        self.pretrain_feat = self.extractor(self.pretrain_images, bn_train=self.bn_train, with_BN=self.with_BN)
        self.logits_pre = self.linear_cls(self.pretrain_feat)
        self.loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pretrain_labels_vec,
                                                                               logits=self.logits_pre,
                                                                               name='loss_pre'))
        self.acc_pre = tf.nn.in_top_k(self.logits_pre, self.pretrain_labels, k=1)
        self.update_ops_pre = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ### (2) episodic training
        ### dataset
        dataset = tf.data.Dataset.from_generator(self.train_episode_generator, (tf.int32))
        dataset = dataset.map(self.make_img_and_lb_from_train_idx, num_parallel_calls=self.num_parallel_calls).batch(1)
        iterator = dataset.make_one_shot_iterator()
        support_images, query_images, query_labels, support_labels, support_classes = iterator.get_next()
        self.support_images = tf.squeeze(support_images, [0])
        self.query_images = tf.squeeze(query_images, [0])
        self.query_labels = tf.squeeze(query_labels, [0])
        self.support_labels = tf.squeeze(support_labels, [0])
        self.support_classes = tf.squeeze(support_classes, [0])
        ### basic operation
        self.support_images_reshape = tf.reshape(self.support_images, shape=[-1]+image_dims) ### shape: [self.n_way*self.n_shot, self.img_size, self.img_size, self.c_dim]
        self.support_feat = self.extractor(self.support_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support = tf.reshape(self.support_feat, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.query_feat = self.extractor(self.query_images, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_query_all, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat, bn_train=self.bn_train, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
        else:
            self.support_class_code = self.support_feat
            self.query_class_code = self.query_feat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_prototypes = tf.reduce_mean(self.support_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro = -tf.norm(self.support_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                               logits=self.logits_pro,
                                                                               name='loss_pro'))
        self.acc_pro = tf.nn.in_top_k(self.logits_pro, self.query_labels, k=1)
        ### hallucination flow
        ### "When there are multiple samples for class y, i.e., K > 1, we simply average the feature vectors
        ###  and take the averaged vector as the prototype of class y" (K. Li, CVPR 2020)
        self.support_ave = tf.reduce_mean(self.support, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        ### [NOTE] make 2 hallucination for each class in the support set
        self.hal_feat_1, self.z_1 = self.build_augmentor(self.support_ave, self.n_way, self.n_shot, self.n_shot+1) ### shape: [self.n_way, self.fc_dim], [self.n_way, self.z_dim]
        self.hallucinated_features_1 = tf.reshape(self.hal_feat_1, shape=[self.n_way, 1, -1]) ### shape: [self.n_way, 1, self.fc_dim]
        self.hal_feat_2, self.z_2 = self.build_augmentor(self.support_ave, self.n_way, self.n_shot, self.n_shot+1, reuse=True) ### shape: [self.n_way, self.fc_dim], [self.n_way, self.z_dim]
        self.hallucinated_features_2 = tf.reshape(self.hal_feat_2, shape=[self.n_way, 1, -1]) ### shape: [self.n_way, 1, self.fc_dim]
        # self.hal_feat = tf.concat((self.hal_feat_1, self.hal_feat_2), axis=0) ### shape=[self.n_way*2, self.fc_dim]
        self.hallucinated_features = tf.concat((self.hallucinated_features_1, self.hallucinated_features_2), axis=1) ### shape: [self.n_way, 2, self.fc_dim]
        self.hal_feat = tf.reshape(self.hallucinated_features, shape=[self.n_way*2, -1]) ### shape: [self.n_way*2, self.fc_dim]
        ### [2020/06/24] Follow AFHN (K. Li, CVPR 2020) to use cosine similarity between hallucinated and query features to compute classification error
        if self.with_pro:
            self.hal_class_code = self.proto_encoder(self.hal_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*2, self.fc_dim]
        else:
            self.hal_class_code = self.hal_feat
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, 2, -1]) ### shape: [self.n_way, 2, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_shot+2, self.fc_dim]
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
        self.acc_pro_aug = tf.nn.in_top_k(cos_sim_ab_exp, query_labels_repeat, k=1)
        ### prototypical network data flow using the augmented support set
        # if self.with_pro:
        #     self.hal_class_code = self.proto_encoder(self.hal_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*2, self.fc_dim]
        # else:
        #     self.hal_class_code = self.hal_feat
        # self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, 2, -1]) ### shape: [self.n_way, 2, self.fc_dim]
        # self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, 3, self.fc_dim]
        # self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        # self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        # self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
        #                                                                            logits=self.logits_pro_aug,
        #                                                                            name='loss_pro_aug'))
        # self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

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
        ### (1) collect all real samples (feature extracted from self.support_images and self.query_images)
        ###     and fake samples (all outputs of self.build_augmentor(): self.hal_feat)
        ###     (for simplicity, only take self.support_feat as real and self.hal_feat as fake)
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
        ### (2) Compute logits
        if self.with_pro:
            self.d_logit_real_1 = self.discriminator_tf(self.proto_encoder(self.feat_real_1, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train)
            self.d_logit_real_2 = self.discriminator_tf(self.proto_encoder(self.feat_real_2, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train, reuse=True)
            self.d_logit_fake_1 = self.discriminator_tf(self.proto_encoder(self.feat_fake_1, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train, reuse=True)
            self.d_logit_fake_2 = self.discriminator_tf(self.proto_encoder(self.feat_fake_2, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train, reuse=True)
        else:
            self.d_logit_real_1 = self.discriminator_tf(self.feat_real_1, bn_train=self.bn_train)
            self.d_logit_real_2 = self.discriminator_tf(self.feat_real_2, bn_train=self.bn_train, reuse=True)
            self.d_logit_fake_1 = self.discriminator_tf(self.feat_fake_1, bn_train=self.bn_train, reuse=True)
            self.d_logit_fake_2 = self.discriminator_tf(self.feat_fake_2, bn_train=self.bn_train, reuse=True)
        ### (3) Compute loss
        ### (Vanilla GAN loss)
        # self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_logit_real),
        #                                                                           logits=self.d_logit_real,
        #                                                                           name='loss_d_real'))
        # self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_logit_real),
        #                                                                           logits=self.d_logit_fake,
        #                                                                           name='loss_d_fake'))
        # self.loss_d_TF = self.loss_d_real + self.loss_d_fake
        # self.loss_g_TF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_logit_fake),
        #                                                                         logits=self.d_logit_fake,
        #                                                                         name='loss_g_TF'))
        ### (WGAN loss)
        self.loss_d_real = tf.reduce_mean(self.d_logit_real_1) + tf.reduce_mean(self.d_logit_real_2)
        self.loss_d_fake = tf.reduce_mean(self.d_logit_fake_1) + tf.reduce_mean(self.d_logit_fake_2)
        self.loss_d_tf = self.loss_d_fake - self.loss_d_real
        self.loss_g_tf = (-1) * (tf.reduce_mean(self.d_logit_fake_1) + tf.reduce_mean(self.d_logit_fake_2))
        epsilon_1 = tf.random_uniform([], 0.0, 1.0)
        x_hat_1 = epsilon_1 * self.s_real + (1 - epsilon_1) * self.hal_feat_1
        # d_hat_1 = self.discriminator_tf(tf.concat((x_hat_1, self.z_1), axis=1), self.bn_train, reuse=True)
        d_hat_1 = self.discriminator_tf(x_hat_1, self.bn_train, reuse=True)
        self.ddx_1 = tf.gradients(d_hat_1, x_hat_1)[0]
        self.ddx_1 = tf.sqrt(tf.reduce_sum(tf.square(self.ddx_1), axis=1))
        self.ddx_1 = tf.reduce_mean(tf.square(self.ddx_1 - 1.0) * self.gp_scale)
        epsilon_2 = tf.random_uniform([], 0.0, 1.0)
        x_hat_2 = epsilon_2 * self.s_real + (1 - epsilon_2) * self.hal_feat_2
        # d_hat_2 = self.discriminator_tf(tf.concat((x_hat_2, self.z_2), axis=1), self.bn_train, reuse=True)
        d_hat_2 = self.discriminator_tf(x_hat_2, self.bn_train, reuse=True)
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
        self.update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if not op in self.update_ops_pre]
        
        ### (3) model parameters for testing
        ### dataset
        dataset_t = tf.data.Dataset.from_generator(self.test_episode_generator, (tf.int32))
        dataset_t = dataset_t.map(self.make_img_and_lb_from_test_idx, num_parallel_calls=self.num_parallel_calls)
        dataset_t = dataset_t.batch(1)
        iterator_t = dataset_t.make_one_shot_iterator()
        support_images_t, query_images_t, query_labels_t, support_classes_t = iterator_t.get_next()
        self.support_images_t = tf.squeeze(support_images_t, [0])
        self.query_images_t = tf.squeeze(query_images_t, [0])
        self.query_labels_t = tf.squeeze(query_labels_t, [0])
        self.support_classes_t = tf.squeeze(support_classes_t, [0])
        ### operation
        self.support_images_reshape_t = tf.reshape(self.support_images_t, shape=[-1]+image_dims)
        self.support_feat_t = self.extractor(self.support_images_reshape_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.support_t = tf.reshape(self.support_feat_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.query_feat_t = self.extractor(self.query_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.query_labels_vec_t = tf.one_hot(self.query_labels_t, self.n_way_t)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code_t = self.proto_encoder(self.support_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
            self.query_class_code_t = self.proto_encoder(self.query_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.support_class_code_t = self.support_feat_t
            self.query_class_code_t = self.query_feat_t
        self.support_encode_t = tf.reshape(self.support_class_code_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.support_prototypes_t = tf.reduce_mean(self.support_encode_t, axis=1)
        self.query_tile_t = tf.reshape(tf.tile(self.query_class_code_t, multiples=[1, self.n_way_t]), [self.n_query_all_t, self.n_way_t, -1])
        self.logits_pro_t = -tf.norm(self.support_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                 logits=self.logits_pro_t,
                                                                                 name='loss_pro_t'))
        self.acc_pro_t = tf.nn.in_top_k(self.logits_pro_t, self.query_labels_t, k=1)
        ### prototypical network data flow with hallucination
        self.support_ave_t = tf.reduce_mean(self.support_t, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.hal_feat_t, _ = self.build_augmentor(self.support_ave_t, self.n_way_t, self.n_shot_t, self.n_aug_t, reuse=True)
        self.hallucinated_features_t = tf.reshape(self.hal_feat_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])
        if self.with_pro:
            self.hal_class_code_t = self.proto_encoder(self.hal_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.hal_class_code_t = self.hal_feat_t
        self.hal_encode_t = tf.reshape(self.hal_class_code_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])
        self.support_aug_encode_t = tf.concat((self.support_encode_t, self.hal_encode_t), axis=1)
        self.support_aug_prototypes_t = tf.reduce_mean(self.support_aug_encode_t, axis=1)
        self.logits_pro_aug_t = -tf.norm(self.support_aug_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_aug_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                     logits=self.logits_pro_aug_t,
                                                                                     name='loss_pro_aug_t'))
        self.acc_pro_aug_t = tf.nn.in_top_k(self.logits_pro_aug_t, self.query_labels_t, k=1)
        ### [2020/06/22] Follow AFHN (K. Li, CVPR 2020) to train a linear classifier on the augmented support set and classify samples from the query set
        ### (1) train on the augmented support set
        ### feed the above self.support_aug_encode_t (use class codes) and the corresponding meta labels (0, 1, ..., self.n_way_t)
        # self.feat_meta_train = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='feat_meta_train')
        # self.lb_meta_train = tf.placeholder(tf.int32, shape=[None], name='lb_meta_train')
        # self.lb_meta_train_vec = tf.one_hot(self.lb_meta_train, self.n_way_t)
        # self.logits_meta_train = self.meta_cls(self.feat_meta_train)
        # self.loss_meta_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.lb_meta_train_vec,
        #                                                                               logits=self.logits_meta_train,
        #                                                                               name='loss_meta_train'))
        # self.acc_meta_train = tf.nn.in_top_k(self.logits_meta_train, self.lb_meta_train, k=1)
        # ### (2) test on the query set
        # ### feed the above self.query_class_code_t (again, use class codes) and the corresponding meta labels (0, 1, ..., self.n_way_t)
        # self.feat_meta_test = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='feat_meta_test')
        # self.lb_meta_test = tf.placeholder(tf.int32, shape=[None], name='lb_meta_test')
        # self.lb_meta_test_vec = tf.one_hot(self.lb_meta_test, self.n_way_t)
        # self.logits_meta_test = self.meta_cls(self.feat_meta_test, reuse=True)
        # self.loss_meta_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.lb_meta_test_vec,
        #                                                                              logits=self.logits_meta_test,
        #                                                                              name='loss_meta_test'))
        # self.acc_meta_test = tf.nn.in_top_k(self.logits_meta_test, self.lb_meta_test, k=1)

        ### [2020/07/01] test if the hallucinated features generated from the same set of noise vectors will be more and more diverse as the training proceeds
        s_test = tf.random_uniform([1, 1, self.fc_dim], 0.0, 1.0, seed=1002)
        self.hal_feat_test, self.z_test = self.build_augmentor_test(s_test, 1, 1, 3, reuse=True)
        squared_h = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.hal_feat_test), axis=1, keep_dims=True)), multiples=[1,self.fc_dim])
        self.hal_feat_test_normalized = self.hal_feat_test / (squared_h + 1e-10)
        

        
        ### (4) data loader and operation for testing class set for visualization
        ### dataset
        image_paths_t = tf.convert_to_tensor(self.test_image_list, dtype=tf.string)
        dataset_pre_t = tf.data.Dataset.from_tensor_slices(image_paths_t)
        dataset_pre_t = dataset_pre_t.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_t = dataset_pre_t.make_one_shot_iterator()
        pretrain_images_t = iterator_pre_t.get_next()
        self.pretrain_images_t = tf.squeeze(pretrain_images_t, [1, 2])
        ### operation
        self.test_feat = self.extractor(self.pretrain_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.test_class_code = self.proto_encoder(self.test_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.test_class_code = self.test_feat

        ### (5) data loader and operation for training class for visualization
        ### dataset
        image_paths_train = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        dataset_pre_train = tf.data.Dataset.from_tensor_slices(image_paths_train)
        dataset_pre_train = dataset_pre_train.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_train = dataset_pre_train.make_one_shot_iterator()
        pretrain_images_train = iterator_pre_train.get_next()
        self.pretrain_images_train = tf.squeeze(pretrain_images_train, [1, 2])
        ### operation
        self.train_feat = self.extractor(self.pretrain_images_train, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.train_class_code = self.proto_encoder(self.train_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.train_class_code = self.train_feat
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_ext = [var for var in self.all_vars if ('ext' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_ext = [var for var in self.trainable_vars if ('ext' in var.name or 'cls' in var.name)]
        self.trainable_vars_d = [var for var in self.trainable_vars if ('discriminator' in var.name)]
        self.trainable_vars_hal = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name or 'cls' in var.name)]
        self.trainable_vars_pro = [var for var in self.trainable_vars if ('ext' in var.name or 'pro' in var.name)]
        # self.trainable_vars_meta = [var for var in self.trainable_vars if ('meta_classifier' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_ext = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_d = [reg for reg in self.all_regs if \
                            ('discriminator' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_pro = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'pro' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        # self.used_regs_meta = [reg for reg in self.all_regs if \
        #                       ('meta_classifier' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops_pre):
            self.opt_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pre+sum(self.used_regs_ext),
                                                                      var_list=self.trainable_vars_ext)
        with tf.control_dependencies(self.update_ops):
            self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.5).minimize(self.loss_d+sum(self.used_regs_d),
                                                                    var_list=self.trainable_vars_d)
            self.opt_hal = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_all+sum(self.used_regs_hal),
                                                                      var_list=self.trainable_vars_hal)
            self.opt_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pro+sum(self.used_regs_pro),
                                                                      var_list=self.trainable_vars_pro)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_all+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
            # self.opt_meta = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
            #                                        beta1=0.5).minimize(self.loss_meta_train+sum(self.used_regs_meta),
            #                                                            var_list=self.trainable_vars_meta)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(var_list=self.all_vars, max_to_keep=1)
        self.saver_ext = tf.train.Saver(var_list=self.all_vars_ext, max_to_keep=1)

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
            # x = linear(x, 1024, add_bias=(~with_BN), name='dense1')
            # if with_BN:
            #     x = batch_norm(x, is_train=bn_train)
            x = linear(x, 1024, add_bias=True, name='dense1')
            x = lrelu(x, name='relu1', leak=0.01)
            x = linear(x, 1, add_bias=True, name='dense2')
            #### It's weird to add sigmoid activation function for WGAN.
            # x = tf.nn.sigmoid(x, name='sigmoid1')
            return x
    
    ## "We implement the generator G as a two-layer MLP, with LeakyReLU activation for the first layer
    ##  and ReLU activation for the second one. The dimension of the hidden layer is 1024." (K. Li, CVPR 2020)
    def hallucinator(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            # x = linear_identity(x, 1024, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            # if with_BN:
            #     x = batch_norm(x, is_train=bn_train, name='bn1')
            x = linear_identity(x, 1024, add_bias=True, name='dense1') ## [-1,self.fc_dim]
            x = lrelu(x, name='relu1')
            # x = linear_identity(x, self.fc_dim, add_bias=(~with_BN), name='dense2') ## [-1,self.fc_dim]
            # if with_BN:
            #     x = batch_norm(x, is_train=bn_train, name='bn2')
            x = linear_identity(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
        return x
    
    def build_augmentor(self, support_ave, n_way, n_shot, n_aug, reuse=False):
        input_z_vec = tf.random_normal([n_way*(n_aug-n_shot), self.z_dim], stddev=self.z_std)
        support_ave_tile = tf.tile(support_ave, multiples=[1, n_aug-n_shot, 1])
        support_ave_tile_reshape = tf.reshape(support_ave_tile, shape=[n_way*(n_aug-n_shot), self.fc_dim]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([support_ave_tile_reshape, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat, input_z_vec
    
    ## [2020/07/01] test if the hallucinated features generated from the same set of noise vectors will be more and more diverse as the training proceeds
    def build_augmentor_test(self, support_ave, n_way, n_shot, n_aug, reuse=False):
        input_z_vec = tf.random_normal([n_way*(n_aug-n_shot), self.z_dim], stddev=self.z_std, seed=1002)
        support_ave_tile = tf.tile(support_ave, multiples=[1, n_aug-n_shot, 1])
        support_ave_tile_reshape = tf.reshape(support_ave_tile, shape=[n_way*(n_aug-n_shot), -1]) ### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([support_ave_tile_reshape, input_z_vec], axis=1) #### shape: [n_way*(n_aug-n_shot), self.fc_dim+self.z_dim]
        hal_feat = self.hallucinator(input_mat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=reuse) #### shape: [n_way*(n_aug-n_shot), self.fc_dim]
        return hal_feat, input_z_vec
    
    def train(self,
              ext_from=None,
              ext_from_ckpt=None,
              num_epoch_pretrain=100,
              lr_start_pre=1e-3,
              lr_decay_pre=0.5,
              lr_decay_step_pre=10,
              num_epoch_noHal=0,
              num_epoch_hal=0,
              num_epoch_joint=100,
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
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ### load or pre-train the feature extractor
        if os.path.exists(ext_from):
            could_load_ext, checkpoint_counter_ext = self.load_ext(ext_from, ext_from_ckpt)
        else:
            print('ext_from: %s not exists, train feature extractor from scratch using training-class images' % ext_from)
            loss_pre = []
            acc_pre = []
            for epoch in range(1, (num_epoch_pretrain+1)):
                ##### Follow AFHN to use an initial learning rate 1e-3 which decays to the half every 10 epochs
                lr_pre = lr_start_pre * lr_decay_pre**((epoch-1)//lr_decay_step_pre)
                loss_pre_batch = []
                acc_pre_batch = []
                for idx in tqdm.tqdm(range(self.nBatches)):
                    _, loss, acc = self.sess.run([self.opt_pre, self.loss_pre, self.acc_pre],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr_pre})
                    loss_pre_batch.append(loss)
                    acc_pre_batch.append(np.mean(acc))
                ##### record training loss for each iteration (instead of each epoch)
                loss_pre.append(np.mean(loss_pre_batch))
                acc_pre.append(np.mean(acc_pre_batch))
                print('Epoch: %d (lr=%f), loss_pre: %f, acc_pre: %f' % (epoch, lr_pre, loss_pre[-1], acc_pre[-1]))
        
        ### episodic training
        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []
        best_test_loss = None
        start_time = time.time()
        num_epoch = num_epoch_noHal + num_epoch_hal + num_epoch_joint
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        cos_sim_h1h2_list = []
        for epoch in range(1, (num_epoch+1)):
            ### visualization
            if epoch % n_ep_per_visualization == 0:
                samples_filename = 'base_ep%03d' % epoch
                ##### extract all training-class features using the current feature extractor
                train_feat_all = []
                train_class_code_all = []
                for ite_feat_ext in range(self.nBatches):
                    train_feat, train_class_code = self.sess.run([self.train_feat, self.train_class_code],
                                                                 feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches - 1 and self.nRemainder > 0:
                        train_feat = train_feat[:self.nRemainder,:]
                        train_class_code = train_class_code[:self.nRemainder,:]
                    train_feat_all.append(train_feat)
                    train_class_code_all.append(train_class_code)
                train_feat_all = np.concatenate(train_feat_all, axis=0)
                train_class_code_all = np.concatenate(train_class_code_all, axis=0)
                ##### run one training eposide and extract hallucinated features and the corresponding absolute class labels
                # x_i_image, x_i, query_images, query_labels, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                x_i_image, x_i, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                                                                                self.support, ##### shape: [self.n_way, self.n_shot, self.fc_dim]
                                                                                # self.query_images,
                                                                                # self.query_labels,
                                                                                self.hallucinated_features, ##### shape: [self.n_way, self.n_aug - self.n_shot, self.fc_dim]
                                                                                self.support_aug_encode, ##### shape: [self.n_way, self.n_aug, self.fc_dim]
                                                                                self.support_classes], ##### shape: [self.n_way]
                                                                               feed_dict={self.bn_train: False})
                ##### (x_i_image, nearest image to x_tilde_i) visualization
                x_dim = 224
                x_tilde_i_class_codes = x_class_codes[:,self.n_shot:,:]
                n_class_plot = self.n_way if self.n_way <= 10 else 10
                img_array = np.empty((3*n_class_plot, x_dim, x_dim, 3), dtype='uint8')
                nearest_class_list = []
                nearest_class_list_using_class_code = []
                for lb_idx in range(n_class_plot):
                    # x_i_image
                    img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                    # nearest image to x_tilde_i
                    nearest_idx = (np.sum(np.abs(train_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+n_class_plot,:] = img
                    nearest_class_list.append(str(self.train_class_list_raw[nearest_idx]))
                    # nearest image to x_tilde_i_class_codes
                    nearest_idx = (np.sum(np.abs(train_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+2*n_class_plot,:] = img
                    nearest_class_list_using_class_code.append(str(self.train_class_list_raw[nearest_idx]))
                subtitle_list = [str(y) for y in y_i[0:n_class_plot]] + nearest_class_list + nearest_class_list_using_class_code
                fig = plot(img_array, 4, n_class_plot, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
                
                ##### real and hallucinated features t-SNE visualization
                seed_and_hal_feat = {}
                for lb_idx in range(self.n_way):
                    abs_class = y_i[lb_idx]
                    seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], train_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i[0:10],
                                 all_labels=sorted(set(self.train_class_list_raw)),
                                 n_shot=self.n_shot,
                                 n_min=self.n_aug,
                                 title='real and hallucinated features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_and_hal_class_codes = {}
                    for lb_idx in range(self.n_way):
                        abs_class = y_i[lb_idx]
                        seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], train_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i[0:10],
                                     all_labels=sorted(set(self.train_class_list_raw)),
                                     n_shot=self.n_shot,
                                     n_min=self.n_aug,
                                     title='real and hallucinated class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            #### training loops
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            if epoch <= num_epoch_noHal:
                ##### train the feature extractor and prototypical network without hallucination
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt_pro, self.loss_pro, self.acc_pro],
                                                 feed_dict={self.bn_train: False,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            elif epoch <= num_epoch_hal:
                ##### freeze feature extractor and train hallucinator only
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    if self.lambda_tf > 0:
                        for i_d_update in range(self.d_per_g):
                            _ = self.sess.run(self.opt_d,
                                              feed_dict={self.bn_train: False,
                                                         self.learning_rate: lr})
                    _, loss, acc = self.sess.run([self.opt_hal, self.loss_all, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: False,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            else:
                ##### Train the whole model
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    if self.lambda_tf > 0:
                        for i_d_update in range(self.d_per_g):
                            _ = self.sess.run(self.opt_d,
                                              feed_dict={self.bn_train: False,
                                                         self.learning_rate: lr})
                    # _, loss, acc, cos_sim_z1z2, cos_sim_s1s2, loss_pro_aug, loss_g_tf, loss_ar = self.sess.run([self.opt, self.loss_all, self.acc_pro_aug, self.cos_sim_z1z2, self.cos_sim_s1s2, self.loss_pro_aug, self.loss_g_tf, self.loss_anti_collapse],
                    _, loss, acc = self.sess.run([self.opt, self.loss_all, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: False,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            
            #### validation
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch,
                                                                          apply_hal=(epoch > num_epoch_noHal),
                                                                          plot_samples=(epoch % n_ep_per_visualization == 0),
                                                                          samples_filename='valid_ep%03d'%epoch)
            loss_test.append(test_loss)
            acc_test.append(test_acc)
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training acc: %f (std: %f), valid loss: %f, valid accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (epoch, lr, loss_train[-1], acc_train[-1], np.std(acc_ite_train), test_loss, test_acc, test_acc_std, n_1_acc))
            # print()
            # print('cos_sim_z1z2:', cos_sim_z1z2)
            # print()
            # print('cos_sim_s1s2:', cos_sim_s1s2)
            # print()
            # print('loss_pro_aug:', loss_pro_aug)
            # print()
            # print('loss_g_tf:', loss_g_tf)
            # print()
            # print('loss_ar:', loss_ar)
            # print()
            if np.isnan(loss_train[-1]):
                print('loss_train become NaN, stop training')
                break

            ### [2020/07/01] test if the hallucinated features generated from the same set of noise vectors will be more and more diverse as the training proceeds
            hal_feat_test_normalized = self.sess.run(self.hal_feat_test_normalized)
            cos_sim_h1h2_list.append(np.sum(hal_feat_test_normalized[0,:] * hal_feat_test_normalized[1,:]))
                
            #### save model if performance has improved
            if best_test_loss is None:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)
            elif test_loss < best_test_loss:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)
        ### keep the last?
        # self.saver.save(self.sess,
        #                 os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
        #                 global_step=epoch)
        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, loss_test, acc_train, acc_test, cos_sim_h1h2_list]
    
    # def run_testing(self, n_ite_per_epoch, apply_hal=False, plot_samples=False, samples_filename='sample'):
    #     loss_ite_test = []
    #     acc_ite_test = []
    #     n_1_acc = 0
    #     for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
    #         if apply_hal and plot_samples and ite == 1:
    #             ##### extract all testing-class features using the current feature extractor
    #             test_feat_all = []
    #             test_class_code_all = []
    #             for ite_feat_ext in range(self.nBatches_test):
    #                 test_feat, test_class_code = self.sess.run([self.test_feat, self.test_class_code],
    #                                                          feed_dict={self.bn_train: False})
    #                 if ite_feat_ext == self.nBatches_test - 1 and self.nRemainder_test > 0:
    #                     test_feat = test_feat[:self.nRemainder_test,:]
    #                     test_class_code = test_class_code[:self.nRemainder_test,:]
    #                 test_feat_all.append(test_feat)
    #                 test_class_code_all.append(test_class_code)
    #             test_feat_all = np.concatenate(test_feat_all, axis=0)
    #             test_class_code_all = np.concatenate(test_class_code_all, axis=0)
    #             ##### run one testing eposide and extract hallucinated features and the corresponding absolute class labels
    #             # loss, acc, x_i_image, x_i, query_images_t, query_labels_t, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.loss_pro_aug_t,
    #             loss, acc, x_i_image, x_i, x_tilde_i, x_class_codes, y_i  = self.sess.run([self.loss_pro_aug_t,
    #                                                                                        self.acc_pro_aug_t,
    #                                                                                        self.support_images_t, ##### shape: [self.n_way_t, self.n_shot_t] + image_dims
    #                                                                                        self.support_t, ##### shape: [self.n_way_t, self.n_shot_t, self.fc_dim]
    #                                                                                        # self.query_images_t,
    #                                                                                        # self.query_labels_t,
    #                                                                                        self.hallucinated_features_t, ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t, self.fc_dim]
    #                                                                                        self.support_aug_encode_t, ##### shape: [self.n_way_t, self.n_aug_t, self.fc_dim]
    #                                                                                        self.support_classes_t], ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t]
    #                                                                                       feed_dict={self.bn_train: False})
    #             ##### (x_i_image, x_j_image, nearest image to x_tilde_i) visualization
    #             x_dim = 224
    #             x_tilde_i_class_codes = x_class_codes[:,self.n_shot_t:,:]
    #             # img_array = np.empty((3*self.n_way_t, x_dim, x_dim, 3), dtype='uint8')
    #             # nearest_class_list = []
    #             # nearest_class_list_using_class_code = []
    #             # for lb_idx in range(self.n_way_t):
    #             #     # x_i_image
    #             #     img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
    #             #     # nearest image to x_tilde_i
    #             #     nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
    #             #     file_path = self.test_image_list[nearest_idx]
    #             #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #             #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
    #             #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             #     img_array[lb_idx+self.n_way_t,:] = img
    #             #     nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
    #             #     # nearest image to x_tilde_i_class_codes
    #             #     nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
    #             #     file_path = self.test_image_list[nearest_idx]
    #             #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #             #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
    #             #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             #     img_array[lb_idx+2*self.n_way_t,:] = img
    #             #     nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
    #             # subtitle_list = [str(y) for y in y_i] + nearest_class_list + nearest_class_list_using_class_code
    #             # fig = plot(img_array, 3, self.n_way_t, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
    #             # plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
    #             n_hal_per_class = self.n_aug_t - self.n_shot_t if self.n_aug_t - self.n_shot_t < 5 else 5
    #             for lb_idx in range(len(y_i[0:10])):
    #                 img_array = np.empty((3*n_hal_per_class, x_dim, x_dim, 3), dtype='uint8')
    #                 nearest_class_list = []
    #                 nearest_class_list_using_class_code = []
    #                 for aug_idx in range(n_hal_per_class):
    #                     img_array[aug_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
    #                     nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,aug_idx,:]), axis=1)).argmin()
    #                     file_path = self.test_image_list[nearest_idx]
    #                     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #                     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
    #                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                     img_array[aug_idx+n_hal_per_class,:] = img
    #                     nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
    #                     nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,aug_idx,:]), axis=1)).argmin()
    #                     file_path = self.test_image_list[nearest_idx]
    #                     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    #                     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
    #                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                     img_array[aug_idx+2*n_hal_per_class,:] = img
    #                     nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
    #                 subtitle_list = [str(y_i[lb_idx]) for _ in range(n_hal_per_class)] + nearest_class_list + nearest_class_list_using_class_code
    #                 fig = plot(img_array, 3, n_hal_per_class, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
    #                 plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+('_%03d' % y_i[lb_idx])+'.png'), bbox_inches='tight')
                
    #             ##### real and hallucinated features t-SNE visualization
    #             seed_and_hal_feat = {}
    #             for lb_idx in range(self.n_way_t):
    #                 abs_class = y_i[lb_idx]
    #                 seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
    #             X_all = None
    #             Y_all = None
    #             for lb in y_i[0:10]:
    #                 idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
    #                 feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], test_feat_all[idx_for_this_lb,:]), axis=0)
    #                 labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
    #                 if X_all is None:
    #                     X_all = feat_for_this_lb
    #                     Y_all = labels_for_this_lb
    #                 else:
    #                     X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
    #                     Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
    #             X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
    #             plot_emb_results(_emb=X_all_emb_TSNE30_1000,
    #                              _labels=Y_all,
    #                              considered_lb=y_i,
    #                              all_labels=sorted(set(self.test_class_list_raw)),
    #                              n_shot=self.n_shot_t,
    #                              n_min=self.n_aug_t,
    #                              title='real and hallucinated features',
    #                              save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

    #             ##### real and hallucinated class codes t-SNE visualization
    #             seed_and_hal_class_codes = {}
    #             for lb_idx in range(self.n_way_t):
    #                 abs_class = y_i[lb_idx]
    #                 seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
    #             X_all = None
    #             Y_all = None
    #             for lb in y_i[0:10]:
    #                 idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
    #                 feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], test_class_code_all[idx_for_this_lb,:]), axis=0)
    #                 labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
    #                 if X_all is None:
    #                     X_all = feat_for_this_lb
    #                     Y_all = labels_for_this_lb
    #                 else:
    #                     X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
    #                     Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
    #             X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
    #             plot_emb_results(_emb=X_all_emb_TSNE30_1000,
    #                              _labels=Y_all,
    #                              considered_lb=y_i,
    #                              all_labels=sorted(set(self.test_class_list_raw)),
    #                              n_shot=self.n_shot_t,
    #                              n_min=self.n_aug_t,
    #                              title='real and hallucinated class codes',
    #                              save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
    #         elif apply_hal:
    #             ##### run one testing eposide with haluucination
    #             loss, acc = self.sess.run([self.loss_pro_aug_t, self.acc_pro_aug_t],
    #                                       feed_dict={self.bn_train: False})
    #         else:
    #             ##### run one testing eposide without haluucination
    #             loss, acc = self.sess.run([self.loss_pro_t, self.acc_pro_t],
    #                                       feed_dict={self.bn_train: False})
    #         loss_ite_test.append(loss)
    #         acc_ite_test.append(np.mean(acc))
    #         if np.mean(acc) > 0.99:
    #             n_1_acc += 1
    #     return np.mean(loss_ite_test), np.mean(acc_ite_test), np.std(acc_ite_test), n_1_acc

class HAL_PN_PoseRef(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 train_path,
                 test_path,
                 label_key='image_labels',
                 bsize=128,
                 n_way=5, ## number of classes in the support set
                 n_shot=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query_all=3, ## number of samples in the query set
                 n_way_t=5, ## number of classes in the support set for testing
                 n_shot_t=1, ## number of samples per class in the support set for testing
                 n_aug_t=5, ## number of samples per class in the augmented support set for testing
                 n_query_all_t=75, ## number of samples in the query set for testing
                 fc_dim=512,
                 img_size=84,
                 c_dim=3,
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
                 d_per_g=5,
                 l2scale=0.001,
                 n_train_class=64,
                 n_test_class=16,
                 with_BN=False,
                 with_pro=False,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 num_parallel_calls=8):
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
        self.img_size = img_size
        self.c_dim = c_dim
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
        self.l2scale = l2scale
        self.n_train_class = n_train_class
        self.n_test_class = n_test_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.num_parallel_calls = num_parallel_calls
        
        ### (n_way, n_shot, n_aug, n_query_all) for testing
        self.n_way_t = n_way_t
        self.n_shot_t = n_shot_t
        self.n_aug_t = n_aug_t
        self.n_query_all_t = n_query_all_t
        
        ### prepare datasets
        self.train_path = train_path
        self.test_path = test_path
        self.label_key = label_key
        self.bsize = bsize
        ### Load all training-class data as training data
        with open(self.train_path, 'r') as reader:
            train_dict = json.loads(reader.read())
        self.train_image_list = train_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_train_class-1}
        self.train_class_list_raw = train_dict[self.label_key]
        all_train_class = sorted(set(self.train_class_list_raw))
        print('original train class labeling:')
        print(all_train_class)
        label_mapping = {}
        for new_lb in range(self.n_train_class):
            label_mapping[all_train_class[new_lb]] = new_lb
        self.train_label_list = np.array([label_mapping[old_lb] for old_lb in self.train_class_list_raw])
        print('new train class labeling:')
        print(sorted(set(self.train_label_list)))
        self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
        self.nRemainder = len(self.train_image_list) % self.bsize
        
        ### Load all testing-class data as testing data
        with open(self.test_path, 'r') as reader:
            test_dict = json.loads(reader.read())
        self.test_image_list = test_dict['image_names']
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
        #### such that all labels become in the range {0, 1, ..., self.n_test_class-1}
        self.test_class_list_raw = test_dict[self.label_key]
        all_test_class = sorted(set(self.test_class_list_raw))
        print('original test class labeling:')
        print(all_test_class)
        label_mapping = {}
        for new_lb in range(self.n_test_class):
            label_mapping[all_test_class[new_lb]] = new_lb
        self.test_label_list = np.array([label_mapping[old_lb] for old_lb in self.test_class_list_raw])
        print('new test class labeling:')
        print(sorted(set(self.test_label_list)))
        self.nBatches_test = int(np.ceil(len(self.test_image_list) / self.bsize))
        self.nRemainder_test = len(self.test_image_list) % self.bsize

        ### [2020/03/21] make candidate indexes for each label
        self.candidate_indexes_each_lb_train = {}
        for lb in range(self.n_train_class):
            self.candidate_indexes_each_lb_train[lb] = [idx for idx in range(len(self.train_label_list)) if self.train_label_list[idx] == lb]
        self.candidate_indexes_each_lb_test = {}
        for lb in range(self.n_test_class):
            self.candidate_indexes_each_lb_test[lb] = [idx for idx in range(len(self.test_label_list)) if self.test_label_list[idx] == lb]
        self.all_train_labels = set(self.train_label_list)
        self.all_test_labels = set(self.test_label_list)

    def make_indices_for_train_episode(self, all_labels_set, candidate_indexes, n_way, n_shot, n_shot_after_aug, n_query_all, n_intra):
        ###### sample n_way classes from the set of training classes
        selected_lbs = np.random.choice(list(all_labels_set), n_way, replace=False)
        ###### sample n_way classes from the set of training classes (exclude 'selected_lbs')
        selected_lbs_pose = np.random.choice(list(all_labels_set - set(selected_lbs)), n_way, replace=False)
        try:
            selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_intra+n_shot+n_query_all//n_way, replace=False)) \
                                for lb_idx in range(n_way)]
            #### (optional) do not balance the number of query samples for all classes in each episode
            # selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_each_lb[lb_idx], replace=False)) \
            #                     for lb_idx in range(n_way)]
            selected_indexes_pose = [list(np.random.choice(candidate_indexes[selected_lbs_pose[lb_idx]], n_shot_after_aug-n_shot, replace=False)) \
                                     for lb_idx in range(n_way)]
        except:
            print('[Training] Skip this episode since there are not enough samples for some label')
        return selected_indexes, selected_indexes_pose

    def make_indices_for_test_episode(self, all_labels_set, candidate_indexes, n_way, n_shot, n_shot_after_aug, n_query_all):
        ###### sample n_way classes from the set of testing classes
        selected_lbs = np.random.choice(list(all_labels_set), n_way, replace=False)
        try:
            selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_all//n_way, replace=False)) \
                                for lb_idx in range(n_way)]
            #### (optional) do not balance the number of query samples for all classes in each episode
            # selected_indexes = [list(np.random.choice(candidate_indexes[selected_lbs[lb_idx]], n_shot+n_query_each_lb[lb_idx], replace=False)) \
            #                     for lb_idx in range(n_way)]
            #### during testing, randomly select pose-ref features, regardless of their lables
            selected_indexes_pose = [list(np.random.choice(len(self.train_label_list), n_shot_after_aug-n_shot, replace=False)) \
                                     for lb_idx in range(n_way)]
        except:
            print('[Training] Skip this episode since there are not enough samples for some label')
        return selected_indexes, selected_indexes_pose

    def train_episode_generator(self):
        while True:
            selected_indexes, selected_indexes_pose = self.make_indices_for_train_episode(self.all_train_labels,
                                                                                          self.candidate_indexes_each_lb_train,
                                                                                          self.n_way,
                                                                                          self.n_shot,
                                                                                          self.n_aug,
                                                                                          self.n_query_all,
                                                                                          self.n_intra)
            yield selected_indexes, selected_indexes_pose

    def test_episode_generator(self):
        while True:
            selected_indexes, selected_indexes_pose = self.make_indices_for_test_episode(self.all_test_labels,
                                                                                         self.candidate_indexes_each_lb_test,
                                                                                         self.n_way_t,
                                                                                         self.n_shot_t,
                                                                                         self.n_aug_t,
                                                                                         self.n_query_all_t)
            yield selected_indexes, selected_indexes_pose
    
    ## image loader for multiclass training
    def _parse_function(self, filename, label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        seed = np.random.randint(0, 2 ** 31 - 1)
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        ori_image_shape = tf.shape(img)
        img = tf.image.random_flip_left_right(img, seed=seed)
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.random_crop(img, ori_image_shape, seed=seed)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return img, label
    
    ## image loader for episodic training
    def _load_img(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        augment_size = int(self.img_size * 8 / 7) #### 224 --> 256; 84 --> 96
        img = tf.image.resize_images(img, [augment_size, augment_size])
        img = tf.image.central_crop(img, 0.875)
        ### https://github.com/tensorpack/tensorpack/issues/789
        img = tf.cast(img, tf.float32) / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        img = (img - image_mean) / image_std
        return tf.expand_dims(tf.expand_dims(img, 0), 0) ### shape: [1, 1, 224, 224, 3]

    def _detransform(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return (img*std+mean)*255

    def make_img_and_lb_from_train_idx(self, selected_indexes, selected_indexes_pose):
        pose_ref_intra_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_intra)], axis=1) for i in range(self.n_way)], axis=0)
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_intra, self.n_intra+self.n_shot)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes[i,j])) for j in range(self.n_intra+self.n_shot, self.n_intra+self.n_shot+self.n_query_all//self.n_way)], axis=1) for i in range(self.n_way)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all//self.n_way])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        pose_ref_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes_pose[i,j])) for j in range(self.n_aug-self.n_shot)], axis=1) for i in range(self.n_way)], axis=0)
        ### additional: support labels for loss_aux (labels AFTER re-numbering: {0, 1, ..., self.n_train_class-1})
        support_labels = [tf.gather(self.train_label_list, selected_indexes[i,0]) for i in range(self.n_way)]
        ### additional: pose-ref labels for loss_gan (labels AFTER re-numbering: {0, 1, ..., self.n_train_class-1})
        pose_ref_labels = [tf.gather(self.train_label_list, selected_indexes_pose[i,0]) for i in range(self.n_way)]
        ### additional: absolute class labels for visualization
        support_classes = [tf.gather(self.train_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way)]
        pose_ref_classes = [tf.gather(self.train_class_list_raw, selected_indexes_pose[i,0]) for i in range(self.n_way)]
        return support_images, query_images, query_labels, pose_ref_images, pose_ref_intra_images, support_labels, pose_ref_labels, support_classes, pose_ref_classes

    def make_img_and_lb_from_test_idx(self, selected_indexes, selected_indexes_pose):
        support_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.concat([tf.concat([self._load_img(tf.gather(self.test_image_list, selected_indexes[i,j])) for j in range(self.n_shot_t, self.n_shot_t+self.n_query_all_t//self.n_way_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        query_images = tf.reshape(query_images, [-1, self.img_size, self.img_size, self.c_dim])
        query_labels = tf.range(self.n_way_t)
        query_labels = tf.reshape(query_labels, [-1, 1])
        query_labels = tf.tile(query_labels, [1, self.n_query_all_t//self.n_way_t])
        query_labels = tf.reshape(query_labels, [-1]) ### create meta label tensor, e.g., [0, 0, ..., 0, 1, 1, ..., 1, ..., 4, 4, ..., 4] for 5-way episodes
        pose_ref_images = tf.concat([tf.concat([self._load_img(tf.gather(self.train_image_list, selected_indexes_pose[i,j])) for j in range(self.n_aug_t-self.n_shot_t)], axis=1) for i in range(self.n_way_t)], axis=0)
        ### additional: absolute class labels for visualization
        support_classes = [tf.gather(self.test_class_list_raw, selected_indexes[i,0]) for i in range(self.n_way_t)]
        pose_ref_classes = [[tf.gather(self.train_class_list_raw, selected_indexes_pose[i,j]) for i in range(self.n_way_t)] for j in range(self.n_aug_t-self.n_shot_t)]
        return support_images, query_images, query_labels, pose_ref_images, support_classes, pose_ref_classes

    def build_model(self):
        ### training parameters
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        
        ### model parameters
        image_dims = [self.img_size, self.img_size, self.c_dim]
        
        ### (1) pre-train the feature extraction network using training-class images
        ### dataset
        image_paths = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(self.train_label_list)
        dataset_pre = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset_pre = dataset_pre.map(self._parse_function,
                                      num_parallel_calls=self.num_parallel_calls).shuffle(len(self.train_image_list)).prefetch(self.bsize).repeat().batch(self.bsize)
        iterator_pre = dataset_pre.make_one_shot_iterator()
        self.pretrain_images, self.pretrain_labels = iterator_pre.get_next()
        ### operation
        self.pretrain_labels_vec = tf.one_hot(self.pretrain_labels, self.n_train_class)
        self.pretrain_feat = self.extractor(self.pretrain_images, bn_train=self.bn_train, with_BN=self.with_BN)
        self.logits_pre = self.linear_cls(self.pretrain_feat)
        self.loss_pre = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.pretrain_labels_vec,
                                                                               logits=self.logits_pre,
                                                                               name='loss_pre'))
        self.acc_pre = tf.nn.in_top_k(self.logits_pre, self.pretrain_labels, k=1)
        self.update_ops_pre = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### (2) episodic training
        ### dataset
        dataset = tf.data.Dataset.from_generator(self.train_episode_generator, (tf.int32, tf.int32))
        dataset = dataset.map(self.make_img_and_lb_from_train_idx, num_parallel_calls=self.num_parallel_calls).batch(1)
        iterator = dataset.make_one_shot_iterator()
        support_images, query_images, query_labels, pose_ref_images, pose_ref_intra_images, support_labels, pose_ref_labels, support_classes, pose_ref_classes = iterator.get_next()
        self.support_images = tf.squeeze(support_images, [0])
        self.query_images = tf.squeeze(query_images, [0])
        self.query_labels = tf.squeeze(query_labels, [0])
        self.pose_ref_images = tf.squeeze(pose_ref_images, [0])
        self.pose_ref_intra_images = tf.squeeze(pose_ref_intra_images, [0])
        self.support_labels = tf.squeeze(support_labels, [0])
        self.pose_ref_labels = tf.squeeze(pose_ref_labels, [0])
        self.support_classes = tf.squeeze(support_classes, [0])
        self.pose_ref_classes = tf.squeeze(pose_ref_classes, [0])
        ### basic operation
        self.support_images_reshape = tf.reshape(self.support_images, shape=[-1]+image_dims) ### shape: [self.n_way*self.n_shot, self.img_size, self.img_size, self.c_dim]
        self.support_feat = self.extractor(self.support_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*self.n_shot, self.fc_dim]
        self.support = tf.reshape(self.support_feat, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.query_feat = self.extractor(self.query_images, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_query_all, self.fc_dim]
        self.query_labels_vec = tf.one_hot(self.query_labels, self.n_way)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code = self.proto_encoder(self.support_feat, bn_train=self.bn_train, with_BN=self.with_BN) #### shape: [self.n_way*self.n_shot, self.fc_dim]
            self.query_class_code = self.proto_encoder(self.query_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_query_all, self.fc_dim]
        else:
            self.support_class_code = self.support_feat
            self.query_class_code = self.query_feat
        self.support_encode = tf.reshape(self.support_class_code, shape=[self.n_way, self.n_shot, -1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_prototypes = tf.reduce_mean(self.support_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.query_tile = tf.reshape(tf.tile(self.query_class_code, multiples=[1, self.n_way]), [self.n_query_all, self.n_way, -1]) #### shape: [self.n_query_all, self.n_way, self.fc_dim]
        self.logits_pro = -tf.norm(self.support_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                               logits=self.logits_pro,
                                                                               name='loss_pro'))
        self.acc_pro = tf.nn.in_top_k(self.logits_pro, self.query_labels, k=1)
        ### hallucination flow
        self.pose_ref_images_reshape = tf.reshape(self.pose_ref_images, shape=[-1]+image_dims) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.img_size, self.img_size, self.c_dim]
        self.pose_ref_feat = self.extractor(self.pose_ref_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref = tf.reshape(self.pose_ref_feat, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_encode_ave = tf.reduce_mean(self.support_encode, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.support_encode_ave_tile = tf.tile(self.support_encode_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_encode_ave_tile_flat = tf.reshape(self.support_encode_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.transformed_pose_ref, self.pose_code = self.transformer(self.pose_ref_feat, self.support_encode_ave_tile_flat, bn_train=self.bn_train, with_BN=self.with_BN) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.hallucinated_features = tf.reshape(self.transformed_pose_ref, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug = tf.concat([self.support, self.hallucinated_features], axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        ### cycle-consistency loss
        if self.with_pro:
            self.pose_ref_class_code = self.proto_encoder(self.pose_ref_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.pose_ref_class_code = self.pose_ref_feat
        self.pose_ref_encode = tf.reshape(self.pose_ref_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.pose_ref_encode_ave = tf.reduce_mean(self.pose_ref_encode, axis=1, keep_dims=True) ### shape: [self.n_way, 1, self.fc_dim]
        self.pose_ref_encode_ave_tile = tf.tile(self.pose_ref_encode_ave, multiples=[1, self.n_aug-self.n_shot, 1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.pose_ref_encode_ave_tile_flat = tf.reshape(self.pose_ref_encode_ave_tile, shape=[-1, self.fc_dim]) ### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.pose_ref_recon, self.transformed_pose_code = self.transformer(self.transformed_pose_ref, self.pose_ref_encode_ave_tile_flat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        self.loss_pose_ref_recon = tf.reduce_mean((self.pose_ref_feat - self.pose_ref_recon)**2)
        self.loss_pose_code_recon = tf.reduce_mean((self.pose_code - self.transformed_pose_code)**2)
        ### reconstruction loss
        self.support_encode_ave_tile2 = tf.tile(self.support_encode_ave, multiples=[1, self.n_shot, 1]) ### shape: [self.n_way, self.n_shot, self.fc_dim]
        self.support_encode_ave_tile2_flat = tf.reshape(self.support_encode_ave_tile2, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_shot, self.fc_dim]
        self.support_recon, _ = self.transformer(self.support_feat, self.support_encode_ave_tile2_flat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.loss_support_recon = tf.reduce_mean((self.support_feat - self.support_recon)**2)
        ### intra-class transformation consistency loss
        ### (if pose-ref are of the same class as the support samples, then the transformed pose-ref should be as close to the pose-ref itself as possible)
        self.pose_ref_intra_images_reshape = tf.reshape(self.pose_ref_intra_images, shape=[-1]+image_dims) ### shape: [self.n_way*self.n_intra, self.img_size, self.img_size, self.c_dim]
        self.pose_ref_intra_feat = self.extractor(self.pose_ref_intra_images_reshape, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*self.n_intra, self.fc_dim]
        self.support_encode_ave_tile3 = tf.tile(self.support_encode_ave, multiples=[1, self.n_intra, 1]) ### shape: [self.n_way, self.n_intra, self.fc_dim]
        self.support_encode_ave_tile3_flat = tf.reshape(self.support_encode_ave_tile3, shape=[-1, self.fc_dim]) ### shape: [self.n_way*self.n_intra, self.fc_dim]
        self.transformed_pose_ref_intra, _ = self.transformer(self.pose_ref_intra_feat, self.support_encode_ave_tile3_flat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way*self.n_intra, self.fc_dim]
        self.loss_intra = tf.reduce_mean((self.pose_ref_intra_feat - self.transformed_pose_ref_intra)**2)
        ### [2020/06/24] Follow AFHN (K. Li, CVPR 2020) to use cosine similarity between hallucinated and query features to compute classification error
        # if self.with_pro:
        #     self.hal_class_code = self.proto_encoder(self.transformed_pose_ref, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        # else:
        #     self.hal_class_code = self.transformed_pose_ref
        # self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        # self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        # squared_a = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.hal_encode), axis=2, keep_dims=True)), multiples=[1, 1, self.fc_dim])
        # normalize_a = (self.hal_encode / squared_a + 1e-10)
        # normalize_a_tile = tf.tile(tf.transpose(normalize_a, perm=[1, 0, 2]), multiples=[self.n_query_all,1,1])
        # query_labels_repeat = tf.reshape(tf.tile(tf.expand_dims(self.query_labels, axis=1), multiples=[1, self.n_aug-self.n_shot]), [-1])
        # query_labels_repeat_vec = tf.one_hot(query_labels_repeat, self.n_way)
        # squared_b = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(self.query_class_code), axis=1, keep_dims=True)), multiples=[1, self.fc_dim])
        # normalize_b = (self.query_class_code / squared_b + 1e-10)
        # normalize_b_repeat = tf.reshape(tf.tile(tf.expand_dims(normalize_b, axis=1), multiples=[1, self.n_way*(self.n_aug-self.n_shot), 1]), [self.n_query_all*(self.n_aug-self.n_shot), self.n_way, -1]) ### shape: [self.n_query_all*(self.n_aug-self.n_shot), self.n_way, self.fc_dim]
        # cos_sim_ab_exp = tf.exp(tf.reduce_sum(normalize_a_tile * normalize_b_repeat, axis=2))
        # p_y = cos_sim_ab_exp / (tf.reduce_sum(cos_sim_ab_exp, axis=1, keep_dims=True) + 1e-10)
        # self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=query_labels_repeat_vec,
        #                                                                            logits=cos_sim_ab_exp,
        #                                                                            name='loss_pro_aug'))
        # self.acc_pro_aug = tf.nn.in_top_k(cos_sim_ab_exp, query_labels_repeat, k=1)
        ### prototypical network data flow using the augmented support set
        if self.with_pro:
            self.hal_class_code = self.proto_encoder(self.transformed_pose_ref, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) #### shape: [self.n_way*(self.n_aug-self.n_shot), self.fc_dim]
        else:
            self.hal_class_code = self.transformed_pose_ref
        self.hal_encode = tf.reshape(self.hal_class_code, shape=[self.n_way, self.n_aug-self.n_shot, -1]) ### shape: [self.n_way, self.n_aug-self.n_shot, self.fc_dim]
        self.support_aug_encode = tf.concat((self.support_encode, self.hal_encode), axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_prototypes = tf.reduce_mean(self.support_aug_encode, axis=1) ### shape: [self.n_way, self.fc_dim]
        self.logits_pro_aug = -tf.norm(self.support_aug_prototypes - self.query_tile, ord='euclidean', axis=2) ### shape: [self.n_query_all, self.n_way]
        self.loss_pro_aug = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec,
                                                                                   logits=self.logits_pro_aug,
                                                                                   name='loss_pro_aug'))
        self.acc_pro_aug = tf.nn.in_top_k(self.logits_pro_aug, self.query_labels, k=1)

        ### [2020/05/19] Follow IDeMe-Net (Z. Chen, CVPR 2019) to use cross entropy loss to train the network to directly classify the augmented support set
        ### (option 1) use the real support set (meaningless?)
        # self.support_labels_repeat = tf.reshape(tf.tile(tf.expand_dims(self.support_labels, axis=1), multiples=[1, self.n_shot]), [-1])
        # self.support_labels_vec = tf.one_hot(self.support_labels_repeat, self.n_train_class)
        # self.logits_support = self.linear_cls(self.support_feat, reuse=True) ### reuse the classifier used to pre-train the backbone
        # self.loss_aux = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.support_labels_vec,
        #                                                                        logits=self.logits_support,
        #                                                                        name='loss_aux'))
        ### (option 2) use the augmented support set
        self.support_aug_feat = tf.concat([self.support, self.hallucinated_features], axis=1) ### shape: [self.n_way, self.n_aug, self.fc_dim]
        self.support_aug_feat = tf.reshape(self.support_aug_feat, shape=[self.n_way*self.n_aug, -1]) ### shape: [self.n_way*self.n_aug, self.fc_dim]
        self.support_labels_repeat = tf.reshape(tf.tile(tf.expand_dims(self.support_labels, axis=1), multiples=[1, self.n_aug]), [-1])
        self.support_labels_vec = tf.one_hot(self.support_labels_repeat, self.n_train_class)
        self.logits_support = self.linear_cls(self.support_aug_feat, reuse=True) ### reuse the classifier used to pre-train the backbone
        self.loss_aux = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.support_labels_vec,
                                                                               logits=self.logits_support,
                                                                               name='loss_aux'))
        ### Deprecated since it tends to overfit to training data (base-class data).
        ### Also, why not use the output of the prototypical network (i.e., self.support_aug_encode) as the input of self.linear_cls()?

        ### [2020/05/23b] Follow DRIT (H.-Y. Lee and H.-Y. Tseng, ECCV 2018) to use a discriminator to encourage encoder_pose() to not encode any class-specific information
        ### for simplicity, use the pose code extracted from pose-ref images
        ### (1) Prepare real and fake labels
        self.pose_ref_labels_repeat = tf.reshape(tf.tile(tf.expand_dims(self.pose_ref_labels, axis=1), multiples=[1, self.n_aug-self.n_shot]), [-1])
        self.labels_real_vec = tf.one_hot(self.pose_ref_labels_repeat, self.n_train_class)
        self.labels_fake_vec = (1/self.n_train_class)*tf.ones_like(self.labels_real_vec)
        ### (2) Compute logits
        self.logits_pose_code = self.discriminator_pose(self.pose_code, bn_train=self.bn_train, with_BN=self.with_BN)
        ### (3) Compute loss
        self.loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_real_vec,
                                                                             logits=self.logits_pose_code,
                                                                             name='loss_d'))
        self.loss_d = self.lambda_gan * self.loss_d
        self.loss_g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_fake_vec,
                                                                             logits=self.logits_pose_code,
                                                                             name='loss_g'))
        ### Deprecated since it tends to degenerate self.encoder_pose() and hence its output tends to become garbage.
        
        ### [2020/06/04] Follow AFHN (K. Li, CVPR 2020) to use a discriminator to make the hallucinated features (self.transformed_pose_ref) more realistic
        ### (1) collect all real samples (feature extracted from self.support_images, self.query_images, self.pose_ref_images, and self.pose_ref_intra_images)
        ###     and fake samples (all outputs of self.transformer(): self.transformed_pose_ref, self.pose_ref_recon, self.support_recon, and self.transformed_pose_ref_intra)
        ###     (for simplicity, only take self.support_feat as real and self.transformed_pose_ref as fake)
        self.feat_real = self.support_feat
        self.feat_fake = self.transformed_pose_ref
        ### (2) Compute logits
        if self.with_pro:
            self.d_logit_real = self.discriminator_tf(self.proto_encoder(self.feat_real, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train)
            self.d_logit_fake = self.discriminator_tf(self.proto_encoder(self.feat_fake, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True), bn_train=self.bn_train, reuse=True)
        else:
            self.d_logit_real = self.discriminator_tf(self.feat_real, bn_train=self.bn_train)
            self.d_logit_fake = self.discriminator_tf(self.feat_fake, bn_train=self.bn_train, reuse=True)
        ### (3) Compute loss
        ### (Vanilla GAN loss)
        # self.loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_logit_real),
        #                                                                           logits=self.d_logit_real,
        #                                                                           name='loss_d_real'))
        # self.loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_logit_real),
        #                                                                           logits=self.d_logit_fake,
        #                                                                           name='loss_d_fake'))
        # self.loss_d_TF = self.loss_d_real + self.loss_d_fake
        # self.loss_g_TF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_logit_fake),
        #                                                                         logits=self.d_logit_fake,
        #                                                                         name='loss_g_TF'))
        ### (WGAN loss)
        self.loss_d_real = tf.reduce_mean(self.d_logit_real)
        self.loss_d_fake = tf.reduce_mean(self.d_logit_fake)
        self.loss_d_tf = self.loss_d_fake - self.loss_d_real
        self.loss_g_tf = -tf.reduce_mean(self.d_logit_fake)
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.pose_ref_feat + (1 - epsilon) * self.transformed_pose_ref
        d_hat = self.discriminator_tf(x_hat, self.bn_train, reuse=True)
        self.ddx = tf.gradients(d_hat, x_hat)[0]
        self.ddx = tf.sqrt(tf.reduce_sum(tf.square(self.ddx), axis=1))
        self.ddx = tf.reduce_mean(tf.square(self.ddx - 1.0) * self.gp_scale)
        self.loss_d_tf = self.loss_d_tf + self.ddx

        ### combine all loss functions
        self.loss_d = self.loss_d + self.lambda_tf * self.loss_d_tf
        self.loss_all = self.lambda_meta * self.loss_pro_aug + \
                        self.lambda_recon * self.loss_support_recon + \
                        self.lambda_consistency * self.loss_pose_ref_recon + \
                        self.lambda_consistency_pose * self.loss_pose_code_recon + \
                        self.lambda_intra * self.loss_intra + \
                        self.lambda_aux * self.loss_aux + \
                        self.lambda_gan * self.loss_g + \
                        self.lambda_tf * self.loss_g_tf
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        self.update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if not op in self.update_ops_pre]

        ### (3) model parameters for testing
        ### dataset
        dataset_t = tf.data.Dataset.from_generator(self.test_episode_generator, (tf.int32, tf.int32))
        dataset_t = dataset_t.map(self.make_img_and_lb_from_test_idx, num_parallel_calls=self.num_parallel_calls)
        dataset_t = dataset_t.batch(1)
        iterator_t = dataset_t.make_one_shot_iterator()
        support_images_t, query_images_t, query_labels_t, pose_ref_images_t, support_classes_t, pose_ref_classes_t = iterator_t.get_next()
        self.support_images_t = tf.squeeze(support_images_t, [0])
        self.query_images_t = tf.squeeze(query_images_t, [0])
        self.query_labels_t = tf.squeeze(query_labels_t, [0])
        self.pose_ref_images_t = tf.squeeze(pose_ref_images_t, [0])
        self.support_classes_t = tf.squeeze(support_classes_t, [0])
        self.pose_ref_classes_t = tf.squeeze(pose_ref_classes_t, [0])
        ### operation
        self.support_images_reshape_t = tf.reshape(self.support_images_t, shape=[-1]+image_dims)
        self.support_feat_t = self.extractor(self.support_images_reshape_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.support_t = tf.reshape(self.support_feat_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.query_feat_t = self.extractor(self.query_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.query_labels_vec_t = tf.one_hot(self.query_labels_t, self.n_way_t)
        ### prototypical network data flow without hallucination
        if self.with_pro:
            self.support_class_code_t = self.proto_encoder(self.support_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
            self.query_class_code_t = self.proto_encoder(self.query_feat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.support_class_code_t = self.support_feat_t
            self.query_class_code_t = self.query_feat_t
        self.support_encode_t = tf.reshape(self.support_class_code_t, shape=[self.n_way_t, self.n_shot_t, -1])
        self.support_prototypes_t = tf.reduce_mean(self.support_encode_t, axis=1)
        self.query_tile_t = tf.reshape(tf.tile(self.query_class_code_t, multiples=[1, self.n_way_t]), [self.n_query_all_t, self.n_way_t, -1])
        self.logits_pro_t = -tf.norm(self.support_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                   logits=self.logits_pro_t,
                                                                                   name='loss_pro_t'))
        self.acc_pro_t = tf.nn.in_top_k(self.logits_pro_t, self.query_labels_t, k=1)
        ### prototypical network data flow with hallucination
        self.pose_ref_images_reshape_t = tf.reshape(self.pose_ref_images_t, shape=[-1]+image_dims)
        self.pose_ref_feat_t = self.extractor(self.pose_ref_images_reshape_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        self.pose_ref_t = tf.reshape(self.pose_ref_feat_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])

        self.support_encode_ave_t = tf.reduce_mean(self.support_encode_t, axis=1, keep_dims=True) ### shape: [self.n_way_t, 1, self.fc_dim]
        self.support_encode_ave_tile_t = tf.tile(self.support_encode_ave_t, multiples=[1, self.n_aug_t-self.n_shot_t, 1]) ### shape: [self.n_way_t, self.n_aug_t-self.n_shot_t, self.fc_dim]
        self.support_encode_ave_tile_flat_t = tf.reshape(self.support_encode_ave_tile_t, shape=[-1, self.fc_dim]) ### shape: [self.n_way_t*(self.n_aug_t-self.n_shot_t), self.fc_dim]
        self.transformed_pose_ref_t, self.pose_code_t = self.transformer(self.pose_ref_feat_t, self.support_encode_ave_tile_flat_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True) ### shape=[self.n_way_t*(self.n_aug_t-self.n_shot_t), self.fc_dim]
        self.hallucinated_features_t = tf.reshape(self.transformed_pose_ref_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1]) ### shape: [self.n_way_t, self.n_aug_t-self.n_shot_t, self.fc_dim]
        self.support_aug_t = tf.concat([self.support_t, self.hallucinated_features_t], axis=1)
        if self.with_pro:
            self.hal_class_code_t = self.proto_encoder(self.transformed_pose_ref_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.hal_class_code_t = self.transformed_pose_ref_t
        self.hal_encode_t = tf.reshape(self.hal_class_code_t, shape=[self.n_way_t, self.n_aug_t-self.n_shot_t, -1])
        self.support_aug_encode_t = tf.concat((self.support_encode_t, self.hal_encode_t), axis=1)
        self.support_aug_prototypes_t = tf.reduce_mean(self.support_aug_encode_t, axis=1)
        self.logits_pro_aug_t = -tf.norm(self.support_aug_prototypes_t - self.query_tile_t, ord='euclidean', axis=2)
        self.loss_pro_aug_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.query_labels_vec_t,
                                                                                     logits=self.logits_pro_aug_t,
                                                                                     name='loss_pro_aug_t'))
        self.acc_pro_aug_t = tf.nn.in_top_k(self.logits_pro_aug_t, self.query_labels_t, k=1)
        
        ### (4) data loader and operation for testing class set for visualization
        ### dataset
        image_paths_t = tf.convert_to_tensor(self.test_image_list, dtype=tf.string)
        dataset_pre_t = tf.data.Dataset.from_tensor_slices(image_paths_t)
        dataset_pre_t = dataset_pre_t.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_t = dataset_pre_t.make_one_shot_iterator()
        pretrain_images_t = iterator_pre_t.get_next()
        self.pretrain_images_t = tf.squeeze(pretrain_images_t, [1, 2])
        ### operation
        self.test_feat = self.extractor(self.pretrain_images_t, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.test_class_code = self.proto_encoder(self.test_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.test_class_code = self.test_feat
        self.test_pose_code = self.encoder_pose(self.test_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)

        ### (5) data loader and operation for training class for visualization
        ### dataset
        image_paths_train = tf.convert_to_tensor(self.train_image_list, dtype=tf.string)
        dataset_pre_train = tf.data.Dataset.from_tensor_slices(image_paths_train)
        dataset_pre_train = dataset_pre_train.map(self._load_img).repeat().batch(self.bsize)
        iterator_pre_train = dataset_pre_train.make_one_shot_iterator()
        pretrain_images_train = iterator_pre_train.get_next()
        self.pretrain_images_train = tf.squeeze(pretrain_images_train, [1, 2])
        ### operation
        self.train_feat = self.extractor(self.pretrain_images_train, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        if self.with_pro:
            self.train_class_code = self.proto_encoder(self.train_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.train_class_code = self.train_feat
        self.train_pose_code = self.encoder_pose(self.train_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_ext = [var for var in self.all_vars if ('ext' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_ext = [var for var in self.trainable_vars if ('ext' in var.name or 'cls' in var.name)]
        self.trainable_vars_d = [var for var in self.trainable_vars if ('discriminator' in var.name)]
        self.trainable_vars_hal = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name or 'cls' in var.name)]
        self.trainable_vars_pro = [var for var in self.trainable_vars if ('ext' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_ext = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_d = [reg for reg in self.all_regs if \
                            ('discriminator' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name or 'cls' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_pro = [reg for reg in self.all_regs if \
                              ('ext' in reg.name or 'pro' in reg.name) and (('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        with tf.control_dependencies(self.update_ops_pre):
            self.opt_pre = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pre+sum(self.used_regs_ext),
                                                                      var_list=self.trainable_vars_ext)
        with tf.control_dependencies(self.update_ops):
            self.opt_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.5).minimize(self.loss_d+sum(self.used_regs_d),
                                                                    var_list=self.trainable_vars_d)
            self.opt_hal = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_all+sum(self.used_regs_hal),
                                                                      var_list=self.trainable_vars_hal)
            self.opt_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_pro+sum(self.used_regs_pro),
                                                                      var_list=self.trainable_vars_pro)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                              beta1=0.5).minimize(self.loss_all+sum(self.used_regs),
                                                                  var_list=self.trainable_vars)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(var_list=self.all_vars, max_to_keep=1)
        self.saver_ext = tf.train.Saver(var_list=self.all_vars_ext, max_to_keep=1)
        
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
    
    ## ResNet-18
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope('ext', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            #### x.shape = [-1, 84, 84, 3] or [-1, 32, 32, 3]
            #### conv1
            with tf.variable_scope('conv1'):
                x = conv2d(x, output_dim=64, k_h=7, k_w=7, d_h=2, d_w=2, add_bias=(~with_BN))
                #### x.shape = [-1, 42, 42, 64] or [-1, 16, 16, 64]
                if with_BN:
                    x = batch_norm(x, is_train=bn_train)
                x = tf.nn.relu(x)
                x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv2_x
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_2')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv3_x
            x = resblk_first(x, out_channel=128, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_2')
            #### x.shape = [-1, 11, 11, 128] or [-1, 4, 4, 128]
            #### conv4_x
            x = resblk_first(x, out_channel=256, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_2')
            #### x.shape = [-1, 6, 6, 256] or [-1, 2, 2, 256]
            #### conv5_x
            x = resblk_first(x, out_channel=512, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv5_2')
            #### x.shape = [-1, 3, 3, 512] or [-1, 1, 1, 512]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 512]
            return x
    
    ## linear classifier
    def linear_cls(self, x, reuse=False):
        with tf.variable_scope('cls', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.n_train_class, add_bias=True, name='dense1')
        return x
    
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
              ext_from=None,
              ext_from_ckpt=None,
              num_epoch_pretrain=100,
              lr_start_pre=1e-3,
              lr_decay_pre=0.5,
              lr_decay_step_pre=10,
              num_epoch_noHal=0,
              num_epoch_hal=0,
              num_epoch_joint=100,
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
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ### load or pre-train the feature extractor
        if os.path.exists(ext_from):
            could_load_ext, checkpoint_counter_ext = self.load_ext(ext_from, ext_from_ckpt)
        else:
            print('ext_from: %s not exists, train feature extractor from scratch using training-class images' % ext_from)
            loss_pre = []
            acc_pre = []
            for epoch in range(1, (num_epoch_pretrain+1)):
                ##### Follow AFHN to use an initial learning rate 1e-3 which decays to the half every 10 epochs
                lr_pre = lr_start_pre * lr_decay_pre**((epoch-1)//lr_decay_step_pre)
                loss_pre_batch = []
                acc_pre_batch = []
                for idx in tqdm.tqdm(range(self.nBatches)):
                    _, loss, acc = self.sess.run([self.opt_pre, self.loss_pre, self.acc_pre],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr_pre})
                    loss_pre_batch.append(loss)
                    acc_pre_batch.append(np.mean(acc))
                ##### record training loss for each iteration (instead of each epoch)
                loss_pre.append(np.mean(loss_pre_batch))
                acc_pre.append(np.mean(acc_pre_batch))
                print('Epoch: %d (lr=%f), loss_pre: %f, acc_pre: %f' % (epoch, lr_pre, loss_pre[-1], acc_pre[-1]))
        
        ### episodic training
        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []
        best_test_loss = None
        start_time = time.time()
        num_epoch = num_epoch_noHal + num_epoch_hal + num_epoch_joint
        n_ep_per_visualization = num_epoch//10 if num_epoch>10 else 1
        for epoch in range(1, (num_epoch+1)):
            #### visualization
            if epoch % n_ep_per_visualization == 0:
                samples_filename = 'base_ep%03d' % epoch
                ##### extract all training-class features using the current feature extractor
                train_feat_all = []
                train_class_code_all = []
                train_pose_code_all = []
                for ite_feat_ext in range(self.nBatches):
                    train_feat, train_class_code, train_pose_code = self.sess.run([self.train_feat, self.train_class_code, self.train_pose_code],
                                                                            feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches - 1 and self.nRemainder > 0:
                        train_feat = train_feat[:self.nRemainder,:]
                        train_class_code = train_class_code[:self.nRemainder,:]
                        train_pose_code = train_pose_code[:self.nRemainder,:]
                    train_feat_all.append(train_feat)
                    train_class_code_all.append(train_class_code)
                    train_pose_code_all.append(train_pose_code)
                train_feat_all = np.concatenate(train_feat_all, axis=0)
                train_class_code_all = np.concatenate(train_class_code_all, axis=0)
                train_pose_code_all = np.concatenate(train_pose_code_all, axis=0)
                ##### run one training eposide and extract hallucinated features and the corresponding absolute class labels
                # x_i_image, x_i, query_images, query_labels, x_j_image, x_tilde_i, x_class_codes, y_i, y_j  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                x_i_image, x_i, x_j_image, x_tilde_i, x_class_codes, y_i, y_j  = self.sess.run([self.support_images, ##### shape: [self.n_way, self.n_shot] + image_dims
                                                                                                self.support, ##### shape: [self.n_way, self.n_shot, self.fc_dim]
                                                                                                # self.query_images,
                                                                                                # self.query_labels,
                                                                                                self.pose_ref_images, ##### shape: [self.n_way, self.n_aug - self.n_shot] + image_dims
                                                                                                self.hallucinated_features, ##### shape: [self.n_way, self.n_aug - self.n_shot, self.fc_dim]
                                                                                                self.support_aug_encode, ##### shape: [self.n_way, self.n_aug, self.fc_dim]
                                                                                                self.support_classes, ##### shape: [self.n_way]
                                                                                                self.pose_ref_classes], ##### shape: [self.n_way, self.n_aug - self.n_shot]
                                                                                               feed_dict={self.bn_train: False})
                ##### (x_i_image, x_j_image, nearest image to x_tilde_i) visualization
                x_dim = 224
                x_tilde_i_class_codes = x_class_codes[:,self.n_shot:,:]
                n_class_plot = self.n_way if self.n_way <= 10 else 10
                img_array = np.empty((4*n_class_plot, x_dim, x_dim, 3), dtype='uint8')
                nearest_class_list = []
                nearest_class_list_using_class_code = []
                for lb_idx in range(n_class_plot):
                    # x_i_image
                    img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                    # x_j_image
                    img_array[lb_idx+n_class_plot,:] = (self._detransform(x_j_image[lb_idx,0,:])).astype(int)
                    # nearest image to x_tilde_i
                    nearest_idx = (np.sum(np.abs(train_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+2*n_class_plot,:] = img
                    nearest_class_list.append(str(self.train_class_list_raw[nearest_idx]))
                    # nearest image to x_tilde_i_class_codes
                    nearest_idx = (np.sum(np.abs(train_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
                    file_path = self.train_image_list[nearest_idx]
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_idx+3*n_class_plot,:] = img
                    nearest_class_list_using_class_code.append(str(self.train_class_list_raw[nearest_idx]))
                subtitle_list = [str(y) for y in y_i[0:n_class_plot]] + [str(y) for y in y_j[0:n_class_plot]] + nearest_class_list + nearest_class_list_using_class_code
                fig = plot(img_array, 4, n_class_plot, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
                
                ##### pose code t-SNE visualization
                indexes_for_plot = [idx for idx in range(len(self.train_class_list_raw)) if self.train_class_list_raw[idx] in y_i[0:10]]
                X_all = train_pose_code_all[indexes_for_plot]
                Y_all = np.array([self.train_class_list_raw[idx] for idx in indexes_for_plot])
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i[0:10],
                                 all_labels=sorted(set(self.train_class_list_raw)),
                                 n_shot=0,
                                 n_min=0,
                                 alpha_for_real=0.5, ##### if only plot real samples, increase its alpha (0.3 --> 0.5)
                                 title='real appearance codes',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_pose_code.png'))
                
                ##### real and hallucinated features t-SNE visualization
                seed_and_hal_feat = {}
                for lb_idx in range(self.n_way):
                    abs_class = y_i[lb_idx]
                    seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], train_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i[0:10],
                                 all_labels=sorted(set(self.train_class_list_raw)),
                                 n_shot=self.n_shot,
                                 n_min=self.n_aug,
                                 title='real and hallucinated features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_and_hal_class_codes = {}
                    for lb_idx in range(self.n_way):
                        abs_class = y_i[lb_idx]
                        seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.train_class_list_raw)) if self.train_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], train_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_aug+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i[0:10],
                                     all_labels=sorted(set(self.train_class_list_raw)),
                                     n_shot=self.n_shot,
                                     n_min=self.n_aug,
                                     title='real and hallucinated class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            #### training loops
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            loss_ite_train = []
            acc_ite_train = []
            if epoch <= num_epoch_noHal:
                ##### train the feature extractor and prototypical network without hallucination
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    _, loss, acc = self.sess.run([self.opt_pro, self.loss_pro, self.acc_pro],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            elif epoch <= num_epoch_hal:
                ##### freeze feature extractor and train hallucinator only
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    if self.lambda_gan > 0 or self.lambda_tf > 0:
                        for i_d_update in range(self.d_per_g):
                            _ = self.sess.run(self.opt_d,
                                              feed_dict={self.bn_train: True,
                                                         self.learning_rate: lr})
                    _, loss, acc = self.sess.run([self.opt_hal, self.loss_all, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            else:
                ##### Train the whole model
                for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                    if self.lambda_gan > 0 or self.lambda_tf > 0:
                        for i_d_update in range(self.d_per_g):
                            _ = self.sess.run(self.opt_d,
                                              feed_dict={self.bn_train: True,
                                                         self.learning_rate: lr})
                    _, loss, acc = self.sess.run([self.opt, self.loss_all, self.acc_pro_aug],
                                                 feed_dict={self.bn_train: True,
                                                            self.learning_rate: lr})
                    loss_ite_train.append(loss)
                    acc_ite_train.append(np.mean(acc))
            loss_train.append(np.mean(loss_ite_train))
            acc_train.append(np.mean(acc_ite_train))
            
            #### validation
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch,
                                                                          apply_hal=(epoch > num_epoch_noHal),
                                                                          plot_samples=(epoch % n_ep_per_visualization == 0),
                                                                          samples_filename='valid_ep%03d'%epoch)
            loss_test.append(test_loss)
            acc_test.append(test_acc)
            print('---- Epoch: %d, learning_rate: %f, training loss: %f, training acc: %f (std: %f), valid loss: %f, valid accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (epoch, lr, loss_train[-1], acc_train[-1], np.std(acc_ite_train), test_loss, test_acc, test_acc_std, n_1_acc))
                
            #### save model if performance has improved
            if best_test_loss is None:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)
            elif test_loss < best_test_loss:
                best_test_loss = test_loss
                self.saver.save(self.sess,
                                os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                global_step=epoch)

        print('time: %4.4f' % (time.time() - start_time))
        return [loss_train, loss_test, acc_train, acc_test]
    
    def run_testing(self, n_ite_per_epoch, apply_hal=False, plot_samples=False, samples_filename='sample'):
        loss_ite_test = []
        acc_ite_test = []
        n_1_acc = 0
        for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
            if apply_hal and plot_samples and ite == 1:
                ##### extract all testing-class features using the current feature extractor
                test_feat_all = []
                test_class_code_all = []
                test_pose_code_all = []
                for ite_feat_ext in range(self.nBatches_test):
                    test_feat, test_class_code, test_pose_code = self.sess.run([self.test_feat, self.test_class_code, self.test_pose_code],
                                                                            feed_dict={self.bn_train: False})
                    if ite_feat_ext == self.nBatches_test - 1 and self.nRemainder_test > 0:
                        test_feat = test_feat[:self.nRemainder_test,:]
                        test_class_code = test_class_code[:self.nRemainder_test,:]
                        test_pose_code = test_pose_code[:self.nRemainder_test,:]
                    test_feat_all.append(test_feat)
                    test_class_code_all.append(test_class_code)
                    test_pose_code_all.append(test_pose_code)
                test_feat_all = np.concatenate(test_feat_all, axis=0)
                test_class_code_all = np.concatenate(test_class_code_all, axis=0)
                test_pose_code_all = np.concatenate(test_pose_code_all, axis=0)
                ##### run one testing eposide and extract hallucinated features and the corresponding absolute class labels
                # loss, acc, query_images_t, query_labels_t, x_i_image, x_i, x_j_image, x_tilde_i, x_class_codes, y_i, y_j  = self.sess.run([self.loss_t,
                loss, acc, x_i_image, x_i, x_j_image, x_tilde_i, x_class_codes, y_i, y_j  = self.sess.run([self.loss_pro_aug_t,
                                                                                                           self.acc_pro_aug_t,
                                                                                                           # self.query_images_t,
                                                                                                           # self.query_labels_t,
                                                                                                           self.support_images_t, ##### shape: [self.n_way_t, self.n_shot_t] + image_dims
                                                                                                           self.support_t, ##### shape: [self.n_way_t, self.n_shot_t, self.fc_dim]
                                                                                                           self.pose_ref_images_t, ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t] + image_dims
                                                                                                           self.hallucinated_features_t, ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t, self.fc_dim]
                                                                                                           self.support_aug_encode_t, ##### shape: [self.n_way_t, self.n_aug_t, self.fc_dim]
                                                                                                           self.support_classes_t, ##### shape: [self.n_way_t]
                                                                                                           self.pose_ref_classes_t], ##### shape: [self.n_way_t, self.n_aug_t - self.n_shot_t]
                                                                                                          feed_dict={self.bn_train: False})
                ##### (x_i_image, x_j_image, nearest image to x_tilde_i) visualization
                x_dim = 224
                x_tilde_i_class_codes = x_class_codes[:,self.n_shot_t:,:]
                # img_array = np.empty((4*self.n_way_t, x_dim, x_dim, 3), dtype='uint8')
                # nearest_class_list = []
                # nearest_class_list_using_class_code = []
                # for lb_idx in range(self.n_way_t):
                #     # x_i_image
                #     img_array[lb_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                #     # x_j_image
                #     img_array[lb_idx+self.n_way_t,:] = (self._detransform(x_j_image[lb_idx,0,:])).astype(int)
                #     # nearest image to x_tilde_i
                #     nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,0,:]), axis=1)).argmin()
                #     file_path = self.test_image_list[nearest_idx]
                #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     img_array[lb_idx+2*self.n_way_t,:] = img
                #     nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
                #     # nearest image to x_tilde_i_class_codes
                #     nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,0,:]), axis=1)).argmin()
                #     file_path = self.test_image_list[nearest_idx]
                #     img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                #     img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     img_array[lb_idx+3*self.n_way_t,:] = img
                #     nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
                # subtitle_list = [str(y) for y in y_i] + [str(y) for y in y_j[0,:]] + nearest_class_list + nearest_class_list_using_class_code
                # fig = plot(img_array, 4, self.n_way_t, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                # plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'.png'), bbox_inches='tight')
                n_hal_per_class = self.n_aug_t - self.n_shot_t if self.n_aug_t - self.n_shot_t < 5 else 5
                for lb_idx in range(len(y_i[0:10])):
                    img_array = np.empty((4*n_hal_per_class, x_dim, x_dim, 3), dtype='uint8')
                    nearest_class_list = []
                    nearest_class_list_using_class_code = []
                    for aug_idx in range(n_hal_per_class):
                        img_array[aug_idx,:] = (self._detransform(x_i_image[lb_idx,0,:])).astype(int)
                        img_array[aug_idx+n_hal_per_class,:] = (self._detransform(x_j_image[lb_idx,aug_idx,:])).astype(int)
                        nearest_idx = (np.sum(np.abs(test_feat_all - x_tilde_i[lb_idx,aug_idx,:]), axis=1)).argmin()
                        file_path = self.test_image_list[nearest_idx]
                        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_array[aug_idx+2*n_hal_per_class,:] = img
                        nearest_class_list.append(str(self.test_class_list_raw[nearest_idx]))
                        nearest_idx = (np.sum(np.abs(test_class_code_all - x_tilde_i_class_codes[lb_idx,aug_idx,:]), axis=1)).argmin()
                        file_path = self.test_image_list[nearest_idx]
                        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_array[aug_idx+3*n_hal_per_class,:] = img
                        nearest_class_list_using_class_code.append(str(self.test_class_list_raw[nearest_idx]))
                    subtitle_list = [str(y_i[lb_idx]) for _ in range(n_hal_per_class)] + [str(y_j[i, lb_idx]) for i in range(n_hal_per_class)] + nearest_class_list + nearest_class_list_using_class_code
                    fig = plot(img_array, 4, n_hal_per_class, x_dim=x_dim, subtitles=subtitle_list, fontsize=10)
                    plt.savefig(os.path.join(self.result_path, self.model_name, 'samples', os.path.splitext(samples_filename)[0]+('_%03d' % y_i[lb_idx])+'.png'), bbox_inches='tight')
                
                ##### pose code t-SNE visualization
                indexes_for_plot = [idx for idx in range(len(self.test_class_list_raw)) if self.test_class_list_raw[idx] in y_i[0:10]]
                X_all = test_pose_code_all[indexes_for_plot]
                Y_all = np.array([self.test_class_list_raw[idx] for idx in indexes_for_plot])
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i,
                                 all_labels=sorted(set(self.test_class_list_raw)),
                                 n_shot=0,
                                 n_min=0,
                                 alpha_for_real=0.5, ##### if only plot real samples, increase its alpha (0.3 --> 0.5)
                                 title='real appearance codes',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_pose_code.png'))
                
                ##### real and hallucinated features t-SNE visualization
                seed_and_hal_feat = {}
                for lb_idx in range(self.n_way_t):
                    abs_class = y_i[lb_idx]
                    seed_and_hal_feat[abs_class] = np.concatenate((x_i[lb_idx,:,:], x_tilde_i[lb_idx,:,:]), axis=0)
                X_all = None
                Y_all = None
                for lb in y_i[0:10]:
                    idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                    feat_for_this_lb = np.concatenate((seed_and_hal_feat[lb], test_feat_all[idx_for_this_lb,:]), axis=0)
                    labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
                    if X_all is None:
                        X_all = feat_for_this_lb
                        Y_all = labels_for_this_lb
                    else:
                        X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                        Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                 _labels=Y_all,
                                 considered_lb=y_i,
                                 all_labels=sorted(set(self.test_class_list_raw)),
                                 n_shot=self.n_shot_t,
                                 n_min=self.n_aug_t,
                                 title='real and hallucinated features',
                                 save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_hal_vis.png'))

                ##### real and hallucinated class codes t-SNE visualization
                if self.with_pro:
                    seed_and_hal_class_codes = {}
                    for lb_idx in range(self.n_way_t):
                        abs_class = y_i[lb_idx]
                        seed_and_hal_class_codes[abs_class] = x_class_codes[lb_idx,:,:]
                    X_all = None
                    Y_all = None
                    for lb in y_i[0:10]:
                        idx_for_this_lb = [i for i in range(len(self.test_class_list_raw)) if self.test_class_list_raw[i] == lb]
                        feat_for_this_lb = np.concatenate((seed_and_hal_class_codes[lb], test_class_code_all[idx_for_this_lb,:]), axis=0)
                        labels_for_this_lb = np.repeat(lb, repeats=self.n_aug_t+len(idx_for_this_lb))
                        if X_all is None:
                            X_all = feat_for_this_lb
                            Y_all = labels_for_this_lb
                        else:
                            X_all = np.concatenate((X_all, feat_for_this_lb), axis=0)
                            Y_all = np.concatenate((Y_all, labels_for_this_lb), axis=0)
                    X_all_emb_TSNE30_1000 = dim_reduction(X_all, 'TSNE', 2, 30, 1000)
                    plot_emb_results(_emb=X_all_emb_TSNE30_1000,
                                     _labels=Y_all,
                                     considered_lb=y_i,
                                     all_labels=sorted(set(self.test_class_list_raw)),
                                     n_shot=self.n_shot_t,
                                     n_min=self.n_aug_t,
                                     title='real and hallucinated class codes',
                                     save_path=os.path.join(self.result_path, self.model_name, 'samples', samples_filename+'_class_code.png'))
            elif apply_hal:
                ##### run one testing eposide with haluucination
                loss, acc = self.sess.run([self.loss_pro_aug_t, self.acc_pro_aug_t],
                                          feed_dict={self.bn_train: False})
            else:
                ##### run one testing eposide without haluucination
                loss, acc = self.sess.run([self.loss_pro_t, self.acc_pro_t],
                                          feed_dict={self.bn_train: False})
            loss_ite_test.append(loss)
            acc_ite_test.append(np.mean(acc))
            if np.mean(acc) > 0.99:
                n_1_acc += 1
        return np.mean(loss_ite_test), np.mean(acc_ite_test), np.std(acc_ite_test), n_1_acc
    
    def inference(self,
                  hal_from=None, ## e.g., model_name (must given)
                  hal_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  label_key='image_labels',
                  n_ite_per_epoch=600):
        ### load previous trained hallucinator
        could_load, checkpoint_counter = self.load(hal_from, hal_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch, apply_hal=False, plot_samples=True, samples_filename='novel')
            print('---- Novel (without hal) ---- novel loss: %f, novel accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (test_loss, test_acc, test_acc_std, n_1_acc))
            test_loss, test_acc, test_acc_std, n_1_acc = self.run_testing(n_ite_per_epoch, apply_hal=True, plot_samples=True, samples_filename='novel')
            print('---- Novel (with hal) ---- novel loss: %f, novel accuracy: %f (std: %f) (%d episodes have acc > 0.99)' % \
                (test_loss, test_acc, test_acc_std, n_1_acc))

    ## for loading the trained hallucinator and prototypical network
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    ## for loading the pre-trained feature extractor
    def load_ext(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_ext.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot(self, samples, n_row, n_col):
        fig = plt.figure(figsize=(n_col*2, n_row*2))
        gs = gridspec.GridSpec(n_row, n_col)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(self.img_size, self.img_size, 3))
        return fig

