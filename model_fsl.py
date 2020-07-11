import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from sklearn.metrics import accuracy_score

from ops import *
from utils import *

import pickle
import tqdm

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

# FSL using base-class features and few-shot novel-class features without hallucination
class FSL(object):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 fc_dim,
                 n_class,
                 n_base_class,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 used_opt='sgd'):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.fc_dim = fc_dim
        self.n_class = n_class
        self.n_base_class = n_base_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
        self.used_opt = used_opt
    
    def build_model(self):
        ### model parameters
        self.features = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_vec = tf.one_hot(self.labels, self.n_class)
        ### training parameters
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### data flow
        self.logits = self.fsl_classifier(self.features)
        
        ### loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_vec,
                                                                           logits=self.logits,
                                                                           name='loss'))
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ### variables and regularizers
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizer
        # with tf.control_dependencies(self.update_ops):
        if self.used_opt == 'sgd':
            self.opt_fsl_cls = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                                            var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'momentum':
            self.opt_fsl_cls = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                          momentum=0.9).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                 var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'adam':
            self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                      beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                          var_list=self.trainable_vars_fsl_cls)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(max_to_keep=1)

        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## The classifier is implemented as a simple 1-layer MLP
    def fsl_classifier(self, input_):
        with tf.variable_scope('fsl_cls', regularizer=l2_regularizer(self.l2scale)):
            dense = linear(input_, self.n_class, add_bias=True, name='dense') ## [-1,self.n_class]
        return dense
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              image_path=None,
              label_key='image_labels',
              n_shot=1,
              n_aug=1, ## minimum number of samples per training class ==> (n_aug - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=1000,
              learning_rate=1e-4,
              num_ite=10000,
              novel_idx=None,
              test_mode=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_base_dict = unpickle(train_base_path)
        n_feat_per_base = int(len(train_base_dict[label_key]) / len(set(train_base_dict[label_key])))
        features_base_train = train_base_dict['features']
        labels_base_train = [int(s) for s in train_base_dict[label_key]]
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict['features']
        labels_novel_train = [int(s) for s in train_novel_dict[label_key]]
        
        ### make label mapping for novel_idx
        label_mapping = {}
        labels_novel_train_raw = labels_novel_train
        all_novel_labels = sorted(set(labels_novel_train_raw))
        for new_lb in range(len(all_novel_labels)):
            label_mapping[all_novel_labels[new_lb]] = new_lb
        
        ### For the training split, use all base samples and randomly selected novel samples.
        if n_shot >= n_aug:
            #### Hallucination not needed
            selected_indexes = []
            for lb in all_novel_labels:
                if novel_idx is None:
                    ##### Randomly select n-shot features from each novel class
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                                if labels_novel_train[idx] == lb]
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot, replace=False)
                else:
                    selected_indexes_per_lb = novel_idx[label_mapping[lb],:n_shot]
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            #### Not allowed in the baseline setting
            assert 0
        
        print('features_novel_final.shape:', features_novel_final.shape)
        print('len(labels_novel_final):', len(labels_novel_final))
        print('features_base_train.shape:', features_base_train.shape)
        print('len(labels_base_train):', len(labels_base_train))
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### main training loop
        n_used_classes = len(all_novel_labels) + self.n_base_class
        n_base_per_batch = int(bsize * (self.n_base_class / n_used_classes))
        n_novel_per_batch = bsize - n_base_per_batch
        loss_train = []
        acc_train = []
        top_n_acc_train = []
        loss_train_for_plot = []
        acc_train_for_plot = []
        for ite in range(num_ite):
            batch_idx_base = np.random.choice(len(labels_base_train), n_base_per_batch, replace=False)
            batch_idx_novel = np.random.choice(len(labels_novel_final), n_novel_per_batch, replace=True)
            batch_features = np.concatenate((features_base_train[batch_idx_base], features_novel_final[batch_idx_novel]), axis=0)
            batch_labels = np.array([labels_base_train[i] for i in batch_idx_base]+[labels_novel_final[i] for i in batch_idx_novel])
            _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                            feed_dict={self.features: batch_features,
                                                       self.labels: batch_labels,
                                                       self.learning_rate: learning_rate})
            loss_train.append(loss)
            y_true = batch_labels
            y_pred = np.argmax(logits, axis=1)
            acc_train.append(accuracy_score(y_true, y_pred))
            best_n = np.argsort(logits, axis=1)[:,-n_top:]
            top_n_acc_train.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            if ite % (num_ite//20) == 0:
                loss_train_for_plot.append(np.mean(loss_train[-(num_ite//20):]))
                acc_train_for_plot.append(np.mean(acc_train[-(num_ite//20):]))
                print('Ite: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                      (ite, loss_train[-1], acc_train[-1], n_top, top_n_acc_train[-1]))
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=ite)
        return [loss_train_for_plot, acc_train_for_plot]
    
    def inference(self,
                  test_novel_path, ## test_novel_feat path (must be specified!)
                  test_base_path=None, ## test_base_feat path (if None: close-world; else: open-world)
                  gen_from=None, ## e.g., model_name (must given)
                  gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  label_key='image_labels',
                  out_path=None,
                  n_top=5, ## top-n accuracy
                  bsize=32):
        ### create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ### load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            #### load testing features
            if test_base_path is not None:
                ##### open-world
                test_base_dict = unpickle(test_base_path)
                test_novel_dict = unpickle(test_novel_path)
                features_test = np.concatenate((test_novel_dict['features'], test_base_dict['features']), axis=0)
                labels_base_test = [int(s) for s in test_base_dict[label_key]]
                labels_novel_test = [int(s) for s in test_novel_dict[label_key]]
                labels_test = labels_novel_test + labels_base_test
                #### compute number of batches
                features_len_all = len(labels_test)
                features_len_novel = len(test_novel_dict[label_key])
                nBatches_test = int(np.ceil(features_len_all / bsize))
            else:
                #### close-world
                test_dict = unpickle(test_novel_path)
                features_test = test_dict['features']
                labels_test = test_dict[label_key]
                #### compute number of batches
                features_len_all = len(labels_test)
                features_len_novel = features_len_all
                nBatches_test = int(np.ceil(features_len_all / bsize))
            
            ### make prediction and compute accuracy            
            loss_test_batch = []
            logits_all = None
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_features = features_test[idx*bsize:(idx+1)*bsize]
                batch_labels = labels_test[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features,
                                                        self.labels: batch_labels,
                                                        self.bn_train: False})
                loss_test_batch.append(loss)
                if logits_all is None:
                    logits_all = logits
                else:
                    logits_all = np.concatenate((logits_all, logits), axis=0)
            all_novel_classes = sorted(set(labels_novel_test))
            all_base_classes = sorted(set(labels_base_test))
            is_novel = np.array([(i in all_novel_classes) for i in range(self.n_class)], dtype=int)
            is_base = np.array([(i in all_base_classes) for i in range(self.n_class)], dtype=int)
            is_all = np.array([(i in all_novel_classes or i in all_base_classes) for i in range(self.n_class)], dtype=int)
            
            #### According to Low-Shot Learning from Imaginary Data (Y.-X. Wang, CVPR, 2018),
            #### we can evaluate the "novel" performance as if we already know that all test examples are coming from novel classes.
            #### Quotation:
            #### "1. The model is given test examples from the novel classes, and is only supposed to pick a label from the novel classes.
            ####     That is, the label space is restricted to C_novel (note that doing so is equivalent to setting mu=1 for prototypical networks...)"
            #### Of course, we also evaluate the "all" performance.
            #### Quotation:
            #### "3. The model is given test examples from both the base and novel classes in equal proportion,
            ####     and the model has to predict labels from the joint label space."
            #### TODO: equal proportion?
            score = np.exp(logits_all) / np.repeat(np.sum(np.exp(logits_all), axis=1, keepdims=True), repeats=self.n_class, axis=1)
            score_novel = score * is_novel
            score_all = score * is_all
            
            y_pred_all = np.argmax(score_all, axis=1)
            acc_test_all = accuracy_score(labels_test, y_pred_all)
            best_n_all = np.argsort(score_all, axis=1)[:,-n_top:]
            top_n_acc_test_all = np.mean([(labels_test[idx] in best_n_all[idx]) for idx in range(features_len_all)])
            
            y_pred_novel = np.argmax(score_novel, axis=1)
            acc_test_novel = accuracy_score(labels_test[0:features_len_novel], y_pred_novel[0:features_len_novel])
            best_n_novel = np.argsort(score_novel, axis=1)[:,-n_top:]
            top_n_acc_test_novel = np.mean([(labels_test[idx] in best_n_novel[idx]) for idx in range(features_len_novel)])
            
            print('test loss: %f, test accuracy: %f, top-%d test accuracy: %f, novel test accuracy: %f, novel top-%d test accuracy: %f' % \
                  (np.mean(loss_test_batch), acc_test_all, n_top, top_n_acc_test_all,
                                             acc_test_novel, n_top, top_n_acc_test_novel))
    
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

class FSL_PN_GAN(FSL):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 fc_dim,
                 n_class,
                 n_base_class,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 used_opt='sgd',
                 z_dim=100,
                 z_std=1.0,
                 with_BN=False,
                 with_pro=False):
        super(FSL_PN_GAN, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         fc_dim,
                                         n_class,
                                         n_base_class,
                                         bnDecay,
                                         epsilon,
                                         l2scale,
                                         used_opt)
        self.z_dim = z_dim
        self.z_std = z_std
        self.with_BN = with_BN
        self.with_pro = with_pro

    def build_model(self):
        ### model parameters
        self.features = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_vec = tf.one_hot(self.labels, self.n_class)
        ### training parameters
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### data flow
        if self.with_pro:
            self.features_encode = self.proto_encoder(self.features, bn_train=self.bn_train, with_BN=self.with_BN)
        else:
            self.features_encode = self.features
        self.logits = self.fsl_classifier(self.features_encode)
        
        ### loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_vec,
                                                                           logits=self.logits,
                                                                           name='loss'))

        ### collect update operations for moving-means and moving-variances for batch normalizations
        # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### Also build the hallucinator (No need to define loss or optimizer since we only need foward-pass)
        self.features_and_noise = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim+self.z_dim], name='features_and_noise')  #### shape: [-1, self.z_dim+self.fc_dim] (e.g., 100+512=612)
        self.hallucinated_features = self.hallucinator(self.features_and_noise, bn_train=self.bn_train, with_BN=self.with_BN)
        
        ### variables and regularizers
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)] ### for loading the trained hallucinator and prototypical network
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        # with tf.control_dependencies(self.update_ops):
        if self.used_opt == 'sgd':
            self.opt_fsl_cls = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                                            var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'momentum':
            self.opt_fsl_cls = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                          momentum=0.9).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                 var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'adam':
            self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                      beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                          var_list=self.trainable_vars_fsl_cls)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(max_to_keep=1)
        ### model saver for loading the trained hallucinator and prototypical network
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)
        
        # print('self.all_vars_hal_pro:')
        # for var in self.all_vars_hal_pro:
        #     print(var.name)
        # print()
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    def proto_encoder(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('pro', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
        return x
    
    def hallucinator(self, x, bn_train, with_BN=False, reuse=False):
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
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              image_path=None,
              label_key='image_labels',
              n_shot=1,
              n_aug=20, ## minimum number of samples per training class ==> (n_aug - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=1000,
              learning_rate=1e-4,
              num_ite=10000,
              fix_seed=False,
              novel_idx=None,
              test_mode=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict['features']
        labels_novel_train = [int(s) for s in train_novel_dict[label_key]]
        train_base_dict = unpickle(train_base_path)
        n_feat_per_base = int(len(train_base_dict[label_key]) / len(set(train_base_dict[label_key])))
        features_base_train = train_base_dict['features']
        labels_base_train = [int(s) for s in train_base_dict[label_key]]

        ### make label mapping for novel_idx
        label_mapping = {}
        labels_novel_train_raw = labels_novel_train
        all_novel_labels = sorted(set(labels_novel_train_raw))
        for new_lb in range(len(all_novel_labels)):
            label_mapping[all_novel_labels[new_lb]] = new_lb
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator
        could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro(hal_from, hal_from_ckpt)

        features_novel_final_dict = {} #### for t-SNE feature visualization
        if n_shot >= n_aug:
            #### Hallucination not needed
            selected_indexes = []
            for lb in all_novel_labels:
                if novel_idx is None:
                    ##### Randomly select n-shot features from each class
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                                if labels_novel_train[idx] == lb]
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot, replace=False)
                else:
                    selected_indexes_per_lb = novel_idx[label_mapping[lb],:n_shot]
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in all_novel_labels:
                if novel_idx is None:
                    ##### Randomly select n-shot features from each class
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                                if labels_novel_train[idx] == lb]
                    if fix_seed:
                        selected_indexes_per_lb = candidate_indexes_per_lb[0:n_shot]
                    else:
                        selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot, replace=False)
                else:
                    selected_indexes_per_lb = novel_idx[label_mapping[lb],:n_shot]
                selected_indexes_novel[lb] = selected_indexes_per_lb
            #### randomly sample n_aug-n_shot features from each class (with replacement) and aggregate them to make hallucination seed
            n_hal = n_aug - n_shot
            features_seed_all = np.empty([n_hal * len(all_novel_labels), self.fc_dim])
            lb_counter = 0
            for lb in all_novel_labels:
                seed_indexes_for_this_lb = np.random.choice(selected_indexes_novel[lb], n_aug-n_shot, replace=True)
                features_seed_all[lb_counter*n_hal:(lb_counter+1)*n_hal,:] = features_novel_train[seed_indexes_for_this_lb]
                lb_counter += 1
            #### make hallucination for all novel labels at once
            if not could_load_hal_pro:
                print('Load hallucinator or mlp linear classifier fail!!!!!!')
                feature_hallucinated_all = features_seed_all
            else:
                input_z = np.random.normal(loc=0.0, scale=self.z_std, size=(n_hal*len(all_novel_labels), self.z_dim))
                features_and_noise = np.concatenate((features_seed_all, input_z), axis=1)
                feature_hallucinated_all = self.sess.run(self.hallucinated_features,
                                                         feed_dict={self.features_and_noise: features_and_noise,
                                                                    self.bn_train: False})
                print('feature_hallucinated_all.shape: %s' % (feature_hallucinated_all.shape,))
            #### combine hallucinated and real features to make the final novel feature set
            features_novel_final = np.empty([n_aug * len(all_novel_labels), self.fc_dim])
            labels_novel_final = []
            lb_counter = 0
            for lb in all_novel_labels:
                real_features_per_lb = features_novel_train[selected_indexes_novel[lb]]
                hal_features_per_lb = feature_hallucinated_all[lb_counter*n_hal:(lb_counter+1)*n_hal]
                features_novel_final[lb_counter*n_aug:(lb_counter+1)*n_aug,:] = \
                    np.concatenate((real_features_per_lb, hal_features_per_lb), axis=0)
                labels_novel_final.extend([lb for _ in range(n_aug)])
                lb_counter += 1
                features_novel_final_dict[lb] = np.concatenate((real_features_per_lb, hal_features_per_lb), axis=0)
        
        print('features_novel_final.shape:', features_novel_final.shape)
        print('len(labels_novel_final):', len(labels_novel_final))
        print('features_base_train.shape:', features_base_train.shape)
        print('len(labels_base_train):', len(labels_base_train))

        ### main training loop
        n_used_classes = len(all_novel_labels) + self.n_base_class
        n_base_per_batch = int(bsize * (self.n_base_class / n_used_classes))
        n_novel_per_batch = bsize - n_base_per_batch
        loss_train = []
        acc_train = []
        top_n_acc_train = []
        for ite in range(num_ite):
            batch_idx_base = np.random.choice(len(labels_base_train), n_base_per_batch, replace=False)
            batch_idx_novel = np.random.choice(len(labels_novel_final), n_novel_per_batch, replace=True)
            if ite == 0:
                print('len(batch_idx_base):', len(batch_idx_base))
                print('len(batch_idx_novel):', len(batch_idx_novel))
            batch_features = np.concatenate((features_base_train[batch_idx_base], features_novel_final[batch_idx_novel]), axis=0)
            batch_labels = np.array([labels_base_train[i] for i in batch_idx_base]+[labels_novel_final[i] for i in batch_idx_novel])
            _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                            feed_dict={self.features: batch_features,
                                                       self.labels: batch_labels,
                                                       self.bn_train: False,
                                                       self.learning_rate: learning_rate})
            loss_train.append(loss)
            y_true = batch_labels
            y_pred = np.argmax(logits, axis=1)
            acc_train.append(accuracy_score(y_true, y_pred))
            best_n = np.argsort(logits, axis=1)[:,-n_top:]
            top_n_acc_train.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            if ite % (num_ite//20) == 0:
                print('Ite: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                      (ite, loss_train[-1], acc_train[-1], n_top, top_n_acc_train[-1]))
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=ite)
        return [loss_train, acc_train, features_novel_final_dict]
    
    ## for loading the trained hallucinator and prototypical network
    def load_hal_pro(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

class FSL_PN_GAN2(FSL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 fc_dim,
                 n_class,
                 n_base_class,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 used_opt='sgd',
                 z_dim=100,
                 z_std=1.0,
                 with_BN=False,
                 with_pro=False):
        super(FSL_PN_GAN2, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          fc_dim,
                                          n_class,
                                          n_base_class,
                                          bnDecay,
                                          epsilon,
                                          l2scale,
                                          used_opt,
                                          z_dim,
                                          z_std,
                                          with_BN,
                                          with_pro)

    def hallucinator(self, x, bn_train, with_BN=False, reuse=False):
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

class FSL_PN_AFHN(FSL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 fc_dim,
                 n_class,
                 n_base_class,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 used_opt='sgd',
                 z_dim=100,
                 z_std=1.0,
                 with_BN=False,
                 with_pro=False):
        super(FSL_PN_AFHN, self).__init__(sess,
                                          model_name,
                                          result_path,
                                          fc_dim,
                                          n_class,
                                          n_base_class,
                                          bnDecay,
                                          epsilon,
                                          l2scale,
                                          used_opt,
                                          z_dim,
                                          z_std,
                                          with_BN,
                                          with_pro)
    
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

class FSL_PN_PoseRef(FSL):
    def __init__(self,
                 sess,
                 model_name,
                 result_path,
                 fc_dim,
                 n_class,
                 n_base_class,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 used_opt='sgd',
                 n_gallery_per_class=0,
                 n_base_lb_per_novel=5,
                 with_BN=False,
                 with_pro=False,
                 use_canonical_gallery=False,
                 n_clusters_per_class=0):
        super(FSL_PN_PoseRef, self).__init__(sess,
                                             model_name,
                                             result_path,
                                             fc_dim,
                                             n_class,
                                             n_base_class,
                                             bnDecay,
                                             epsilon,
                                             l2scale,
                                             used_opt)
        self.n_gallery_per_class = n_gallery_per_class
        if n_base_lb_per_novel > 0:
            self.n_base_lb_per_novel = n_base_lb_per_novel
        else:
            self.n_base_lb_per_novel = self.n_base_class
        self.with_BN = with_BN
        self.with_pro = with_pro
        self.use_canonical_gallery = use_canonical_gallery
        self.n_clusters_per_class = n_clusters_per_class
    
    def build_model(self):
        ### model parameters
        self.features = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_vec = tf.one_hot(self.labels, self.n_class)
        ### training parameters
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### data flow
        if self.with_pro:
            self.features_encode = self.proto_encoder(self.features, bn_train=self.bn_train, with_BN=self.with_BN)
        else:
            self.features_encode = self.features
        self.logits = self.fsl_classifier(self.features_encode)
        
        ### loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_vec,
                                                                           logits=self.logits,
                                                                           name='loss'))

        ### collect update operations for moving-means and moving-variances for batch normalizations
        # self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### Also build the hallucinator (No need to define loss or optimizer since we only need foward-pass)
        self.real_class_codes_ave_tile = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='real_class_codes_ave_tile')
        self.pose_feat = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='pose_feat')
        self.hallucinated_features, _ = self.transformer(self.pose_feat, self.real_class_codes_ave_tile, bn_train=self.bn_train, with_BN=self.with_BN)

        ### Data flow to extract class codes and pose codes for visualization
        self.novel_feat = tf.placeholder(tf.float32, shape=[None, self.fc_dim], name='novel_feat')
        if self.with_pro:
            self.novel_code_class = self.proto_encoder(self.novel_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        else:
            self.novel_code_class = self.novel_feat
        self.novel_code_pose = self.encoder_pose(self.novel_feat, bn_train=self.bn_train, with_BN=self.with_BN, reuse=True)
        
        ### variables and regularizers
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)] ### for loading the trained hallucinator and prototypical network
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        # with tf.control_dependencies(self.update_ops):
        if self.used_opt == 'sgd':
            self.opt_fsl_cls = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                                            var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'momentum':
            self.opt_fsl_cls = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                          momentum=0.9).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                                 var_list=self.trainable_vars_fsl_cls)
        elif self.used_opt == 'adam':
            self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                      beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                          var_list=self.trainable_vars_fsl_cls)
        
        ### model saver (keep the best checkpoint)
        self.saver = tf.train.Saver(max_to_keep=1)
        ### model saver for loading the trained hallucinator and prototypical network
        self.saver_hal_pro = tf.train.Saver(var_list=self.all_vars_hal_pro, max_to_keep=1)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    def proto_encoder(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('pro', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
        return x
    
    def encoder_pose(self, x, bn_train, with_BN=False, reuse=False):
        with tf.variable_scope('hal_enc_pose', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
            return x
    
    ## Add a transformer to encode the pose_ref into seed class
    def transformer(self, input_pose, input_class_code, bn_train, with_BN=False, epsilon=1e-5, reuse=False):
        ### [2020/05/21] take average of class codes AFTER proto_encoder()
        code_pose = self.encoder_pose(input_pose, bn_train=bn_train, with_BN=with_BN, reuse=reuse)
        x = tf.concat([input_class_code, code_pose], axis=1)
        with tf.variable_scope('hal_tran', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            x = linear(x, self.fc_dim, add_bias=(~with_BN), name='dense1') ## [-1,self.fc_dim]
            if with_BN:
                x = batch_norm(x, is_train=bn_train)
            x = tf.nn.relu(x, name='relu1')
            x = linear(x, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            x = tf.nn.relu(x, name='relu2')
            return x, code_pose
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              image_path=None,
              label_key='image_labels',
              n_shot=1,
              n_aug=20, ## minimum number of samples per training class ==> (n_aug - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=1000,
              learning_rate=1e-4,
              num_ite=10000,
              fix_seed=False,
              novel_idx=None,
              test_mode=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict['features']
        labels_novel_train = [int(s) for s in train_novel_dict[label_key]]
        fnames_novel_train = train_novel_dict['image_names']
        train_base_dict = unpickle(train_base_path)
        n_feat_per_base = int(len(train_base_dict[label_key]) / len(set(train_base_dict[label_key])))
        features_base_train = train_base_dict['features']
        labels_base_train = [int(s) for s in train_base_dict[label_key]]
        
        ### make label mapping for novel_idx
        label_mapping = {}
        labels_novel_train_raw = labels_novel_train
        all_novel_labels = sorted(set(labels_novel_train_raw))
        for new_lb in range(len(all_novel_labels)):
            label_mapping[all_novel_labels[new_lb]] = new_lb
        
        ### make the gallery set
        train_base_dict_all = unpickle(os.path.join(os.path.dirname(train_base_path), 'base_train_feat'))
        features_base_train_all = train_base_dict_all['features']
        labels_base_train_all = [int(s) for s in train_base_dict_all[label_key]]
        fnames_base_train_all = train_base_dict_all['image_names']
        if self.n_gallery_per_class > 0:
            ### load the index array for the gallery set
            if self.use_canonical_gallery:
                gallery_index_path = os.path.join(os.path.dirname(train_base_path), 'gallery_indexes_canonical_%d.npy' % self.n_gallery_per_class)
            elif self.n_clusters_per_class > 0:
                n_gallery_per_cluster = self.n_gallery_per_class // self.n_clusters_per_class
                gallery_index_path = os.path.join(os.path.dirname(train_base_path), 'gallery_indexes_cluster%dtop%d.npy' % (self.n_clusters_per_class, n_gallery_per_cluster))
            else:
                gallery_index_path = os.path.join(os.path.dirname(train_base_path), 'gallery_indexes_%d.npy' % self.n_gallery_per_class)
            if os.path.exists(gallery_index_path):
                gallery_index_array = np.load(gallery_index_path, allow_pickle=True)
                features_base_gallery = features_base_train_all[gallery_index_array]
                labels_base_gallery = [labels_base_train_all[idx] for idx in range(len(labels_base_train)) if idx in gallery_index_array]
                fnames_base_gallery = [fnames_base_train_all[idx] for idx in range(len(labels_base_train)) if idx in gallery_index_array]
                # print('labels_base_gallery:', labels_base_gallery)
            else:
                print('Load gallery_index_array fail ==> use the whole base-class dataset as the gallery set')
                features_base_gallery = features_base_train_all
                labels_base_gallery = labels_base_train_all
                fnames_base_gallery = fnames_base_train_all
        else:
            features_base_gallery = features_base_train_all
            labels_base_gallery = labels_base_train_all
            fnames_base_gallery = fnames_base_train_all
        candidate_indexes_each_lb_gallery = {}
        for lb in sorted(set(labels_base_gallery)):
            candidate_indexes_each_lb_gallery[lb] = [idx for idx in range(len(labels_base_gallery)) if labels_base_gallery[idx] == lb]
            # print('len(candidate_indexes_each_lb_gallery[%d]): %d' % (lb, len(candidate_indexes_each_lb_gallery[lb])))
        # if self.n_gallery_per_class > 0:
        #     print('candidate_indexes_each_lb_gallery:', candidate_indexes_each_lb_gallery)
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator
        could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro(hal_from, hal_from_ckpt)
        
        #### -------------------------------
        #### for t-SNE feature visualization
        features_novel_final_dict = {}
        pose_feat_dict = {}
        #### -------------------------------
        if n_shot >= n_aug:
            #### Hallucination not needed
            selected_indexes = []
            for lb in all_novel_labels:
                if novel_idx is None:
                    ##### Randomly select n-shot features from each class
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                                if labels_novel_train[idx] == lb]
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot, replace=False)
                else:
                    selected_indexes_per_lb = novel_idx[label_mapping[lb],:n_shot]
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            fnames_novel = []
            for lb in all_novel_labels:
                if novel_idx is None:
                    ##### Randomly select n-shot features from each class
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                                if labels_novel_train[idx] == lb]
                    if fix_seed:
                        selected_indexes_per_lb = candidate_indexes_per_lb[0:n_shot]
                    else:
                        selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot, replace=False)
                else:
                    selected_indexes_per_lb = novel_idx[label_mapping[lb],:n_shot]
                selected_indexes_novel[lb] = selected_indexes_per_lb
                fnames_novel.append(fnames_novel_train[selected_indexes_per_lb[0]])
            #### Make the "real_features" array
            n_hal = n_aug - n_shot
            real_features = np.empty([len(all_novel_labels), n_shot, self.fc_dim])
            lb_counter = 0
            for lb in all_novel_labels:
                real_features[lb_counter,:,:] = features_novel_train[selected_indexes_novel[lb]]
                lb_counter += 1
            #### Use all n_shot features from each novel class (BEFORE the class encoder) to make "real_feat_ave" to find the nearest base classes
            real_feat_ave = np.mean(real_features, axis=1) #### shape: [len(all_novel_labels), self.fc_dim] (since we didn't set keepdims=True)
            #### Use all n_shot features from each novel class (AFTER the class encoder) to make "real_class_codes_ave_tile" for hallucination
            real_class_codes = self.sess.run(self.features_encode,
                                             feed_dict={self.features: np.reshape(real_features, [-1, self.fc_dim]), #### shape: [len(all_novel_labels) * n_shot, self.fc_dim]
                                                        self.bn_train: False})
            real_class_codes_reshape = np.reshape(real_class_codes, [len(all_novel_labels), n_shot, self.fc_dim])
            real_class_codes_ave = np.mean(real_class_codes_reshape, axis=1) #### shape: [len(all_novel_labels), self.fc_dim] (since we didn't set keepdims=True)
            real_class_codes_ave_tile = np.repeat(real_class_codes_ave, repeats=n_hal, axis=0) #### shape: [len(all_novel_labels) * n_hal, self.fc_dim]

            #### [2020/05/05]
            #### (1) compute feature average for each base class
            feat_ave = {}
            for lb in sorted(set(labels_base_gallery)):
                candidate_indexes_each_lb = [idx for idx in range(len(labels_base_gallery)) if labels_base_gallery[idx] == lb]
                feat_ave[lb] = np.mean(features_base_gallery[candidate_indexes_each_lb], axis=0)
            #### (2) use the above feature average to compute the closest class list for each novel class
            closest_class_dict = {}
            lb_counter = 0
            for lb in all_novel_labels:
                feat_ave_this = real_feat_ave[lb_counter]
                ##### the closest class list for each novel class is a list of tuple
                temp_list = []
                for lb_other in sorted(set(labels_base_gallery)):
                    temp_list.append((lb_other, np.sum(np.abs(feat_ave_this - feat_ave[lb_other]))))
                closest_class_dict[lb] = sorted(temp_list, key=lambda x: x[1])
                lb_counter += 1
            
            #### choose the class corresponding to the nearest feature for each seed
            selected_indexes = {}
            fnames_pose_feat = []
            for lb in all_novel_labels:
                lbs_for_pose_ref = [closest_class_dict[lb][idx][0] for idx in range(self.n_base_lb_per_novel)]
                candidate_indexes_for_this_novel = [item for sublist in [candidate_indexes_each_lb_gallery[i] for i in lbs_for_pose_ref] for item in sublist]
                selected_indexes[lb] = list(np.random.choice(candidate_indexes_for_this_novel, n_hal, replace=False))
                fnames_pose_feat.append(fnames_base_gallery[selected_indexes[lb][0]])
                pose_feat_dict[lb] = features_base_gallery[selected_indexes[lb]]
            pose_feat = np.concatenate([pose_feat_dict[lb] for lb in all_novel_labels])

            # if self.n_gallery_per_class > 0:
            #     ##### choose the class corresponding to the nearest feature for each seed
            #     lbs_for_pose_ref = [labels_base_gallery[(np.sum(np.abs(features_base_gallery - real_feat_ave[lb_idx]), axis=1)).argmin()] \
            #                         for lb_idx in range(len(all_novel_labels))]
            #     selected_indexes_each_pose_lb = [list(np.random.choice(candidate_indexes_each_lb_gallery[lbs_for_pose_ref[lb_idx]], n_hal, replace=True)) \
            #                                      for lb_idx in range(len(all_novel_labels))]
            #     pose_feat = np.concatenate([features_base_gallery[selected_indexes_each_pose_lb[lb_idx]] for lb_idx in range(len(all_novel_labels))])
            # else:
            #     ##### randomly sample base-class features from the gallery set to make pose reference
            #     selected_indexes_for_pose_ref = np.random.choice(features_base_gallery.shape[0], n_hal * len(all_novel_labels), replace=True)
            #     pose_feat = features_base_gallery[selected_indexes_for_pose_ref]
            #### make hallucination for all novel labels at once
            if not could_load_hal_pro:
                print('Load hallucinator or mlp linear classifier fail!!!!!!')
                ##### Do nothing, the program should be stopped
                # feature_hallucinated_all = real_feat_ave_tile
            else:
                feature_hallucinated_all = self.sess.run(self.hallucinated_features,
                                                         feed_dict={self.real_class_codes_ave_tile: real_class_codes_ave_tile,
                                                                    self.pose_feat: pose_feat,
                                                                    self.bn_train: False})
                print('feature_hallucinated_all.shape: %s' % (feature_hallucinated_all.shape,))
            n_novel_plot = 10
            idx_list_for_plot = np.random.choice(len(all_novel_labels), n_novel_plot, replace=False)
            fnames_hal_nearest = []
            for lb_idx in idx_list_for_plot:
                fnames_hal_nearest.append(fnames_novel_train[(np.sum(np.abs(features_novel_train - feature_hallucinated_all[lb_idx*n_hal]), axis=1)).argmin()])
            #### combine hallucinated and real features to make the final novel feature set
            features_novel_final = np.empty([n_aug * len(all_novel_labels), self.fc_dim])
            labels_novel_final = []
            lb_counter = 0
            for lb in all_novel_labels:
                real_features_per_lb = features_novel_train[selected_indexes_novel[lb]]
                hal_features_per_lb = feature_hallucinated_all[lb_counter*n_hal:(lb_counter+1)*n_hal]
                features_novel_final[lb_counter*n_aug:(lb_counter+1)*n_aug,:] = \
                    np.concatenate((real_features_per_lb, hal_features_per_lb), axis=0)
                labels_novel_final.extend([lb for _ in range(n_aug)])
                features_novel_final_dict[lb] = np.concatenate((real_features_per_lb, hal_features_per_lb), axis=0)
                lb_counter += 1
            
            #### Plot some hallucination examples if not in test_mode (if in test_mode, we have visualize() in train_fsl_infer.py)
            if not test_mode:
                x_dim = 84
                img_array = np.empty((3*n_novel_plot, x_dim, x_dim, 3), dtype='uint8')
                lb_counter = 0
                for lb_idx in idx_list_for_plot:
                    # fnames_novel
                    file_path = os.path.join(image_path, fnames_novel[lb_idx])
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_counter,:] = img
                    # fnames_pose_feat
                    file_path = os.path.join(image_path, fnames_pose_feat[lb_idx])
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_counter+n_novel_plot,:] = img
                    # fnames_hal_nearest
                    file_path = os.path.join(image_path, fnames_hal_nearest[lb_counter])
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (x_dim, x_dim), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_array[lb_counter+2*n_novel_plot,:] = img
                    lb_counter += 1
                fig = plot(img_array, 3, n_novel_plot, x_dim=x_dim)
                plt.savefig(os.path.join(self.result_path, self.model_name, 'samples.png'), bbox_inches='tight')

        print('features_novel_final.shape:', features_novel_final.shape)
        print('len(labels_novel_final):', len(labels_novel_final))
        print('features_base_train.shape:', features_base_train.shape)
        print('len(labels_base_train):', len(labels_base_train))

        ### main training loop
        n_used_classes = len(all_novel_labels) + self.n_base_class
        n_base_per_batch = int(bsize * (self.n_base_class / n_used_classes))
        n_novel_per_batch = bsize - n_base_per_batch
        print('n_base_per_batch:', n_base_per_batch)
        print('n_novel_per_batch:', n_novel_per_batch)
        loss_train = []
        acc_train = []
        top_n_acc_train = []
        for ite in range(num_ite):
            batch_idx_base = np.random.choice(len(labels_base_train), n_base_per_batch, replace=False)
            batch_idx_novel = np.random.choice(len(labels_novel_final), n_novel_per_batch, replace=True)
            batch_features = np.concatenate((features_base_train[batch_idx_base], features_novel_final[batch_idx_novel]), axis=0)
            batch_labels = np.array([labels_base_train[i] for i in batch_idx_base]+[labels_novel_final[i] for i in batch_idx_novel])
            _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                            feed_dict={self.features: batch_features,
                                                       self.labels: batch_labels,
                                                       self.bn_train: False,
                                                       self.learning_rate: learning_rate})
            loss_train.append(loss)
            y_true = batch_labels
            y_pred = np.argmax(logits, axis=1)
            acc_train.append(accuracy_score(y_true, y_pred))
            best_n = np.argsort(logits, axis=1)[:,-n_top:]
            top_n_acc_train.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            if ite % (num_ite//20) == 0:
                print('Ite: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                      (ite, loss_train[-1], acc_train[-1], n_top, top_n_acc_train[-1]))

        
        ### [20181025] We are not running validation during FSL training since it is meaningless.
        ### Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=ite)

        if test_mode:        
            ### Get the class codes of all seed and hal features
            features_novel_embeded_dict = {}
            features_novel_final_array = np.empty([len(all_novel_labels), n_aug, self.fc_dim])
            lb_counter = 0
            for lb in all_novel_labels:
                features_novel_final_array[lb_counter,:,:] = features_novel_final_dict[lb]
                lb_counter += 1
            features_novel_embeded_array = self.sess.run(self.novel_code_class,
                                                         feed_dict={self.novel_feat: np.reshape(features_novel_final_array, [-1, self.fc_dim]), #### shape: [len(all_novel_labels) * n_aug, self.fc_dim]
                                                                    self.bn_train: False})
            features_novel_embeded_array_reshape = np.reshape(features_novel_embeded_array, [len(all_novel_labels), n_aug, self.fc_dim])
            lb_counter = 0
            for lb in all_novel_labels:
                features_novel_embeded_dict[lb] = features_novel_embeded_array_reshape[lb_counter,:,:]
                lb_counter += 1
        
            ### encode novel features using the two encoders for visualization
            novel_code_class_all = []
            novel_code_pose_all = []
            nBatches_novel = int(np.ceil(features_novel_train.shape[0] / bsize))
            for idx in tqdm.tqdm(range(nBatches_novel)):
                batch_features = features_novel_train[idx*bsize:(idx+1)*bsize]
                novel_code_class, novel_code_pose = self.sess.run([self.novel_code_class, self.novel_code_pose],
                                                                  feed_dict={self.novel_feat: batch_features,
                                                                             self.bn_train: False})
                novel_code_class_all.append(novel_code_class)
                novel_code_pose_all.append(novel_code_pose)
            novel_code_class_all = np.concatenate(novel_code_class_all, axis=0)
            novel_code_pose_all = np.concatenate(novel_code_pose_all, axis=0)
            return [loss_train, acc_train, features_novel_final_dict, pose_feat_dict, features_novel_embeded_dict, novel_code_class_all, novel_code_pose_all]
        else:
            return [loss_train, acc_train, features_novel_final_dict, pose_feat_dict]
    
    ## for loading the trained hallucinator and prototypical network
    def load_hal_pro(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0