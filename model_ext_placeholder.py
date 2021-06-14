import os, re, time, glob
import numpy as np
import json
import tensorflow as tf

from ops import *
from utils import *

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

class BasicCls(object):
    def __init__(self,
                 sess,
                 model_name='BasicCls',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        self.train_path = train_path
        self.test_path = test_path
        self.label_key = label_key
        self.num_epoch = num_epoch
        self.bsize = bsize
        self.bsize_test = bsize * 4
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.img_size = img_size
        self.c_dim = c_dim
        self.n_class = n_class
        self.fc_dim = fc_dim
        self.alpha_center = alpha_center
        self.lambda_center = lambda_center
        self.used_opt = used_opt
        self.use_aug = use_aug
        self.with_BN = with_BN

        self.center_crop = center_crop
        self.aug_size = int(self.img_size * 8 / 7)
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        ### prepare dataset file list
        ### (1) load train and test
        if os.path.exists(self.test_path):
            with open(self.train_path, 'r') as reader:
                train = json.loads(reader.read())
            self.train_image_list = train['image_names']
            #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
            #### such that all labels become {0, 1, 2, ..., self.n_class}
            train_label_list_raw = train[self.label_key]
            all_train_class = sorted(set(train_label_list_raw))
            print('original labeling:')
            print(all_train_class)
            label_mapping = {}
            for new_lb in range(self.n_class):
                label_mapping[all_train_class[new_lb]] = new_lb
            self.train_label_list = np.array([label_mapping[old_lb] for old_lb in train_label_list_raw])
            print('new labeling:')
            print(sorted(set(self.train_label_list)))
            self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
            with open(self.test_path, 'r') as reader:
                test = json.loads(reader.read())
            self.test_image_list = test['image_names']
            #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
            #### such that all labels become 0~158
            test_label_list_raw = test[self.label_key]
            all_test_class = sorted(set(test_label_list_raw))
            label_mapping = {}
            for new_lb in range(self.n_class):
                label_mapping[all_test_class[new_lb]] = new_lb
            self.test_label_list = np.array([label_mapping[old_lb] for old_lb in test_label_list_raw])
            self.nBatches_test = int(np.ceil(len(self.test_image_list) / self.bsize_test))
        ### (2) load base only
        else:
            with open(self.train_path, 'r') as reader:
                base = json.loads(reader.read())
            base_image_list = base['image_names']
            #### Make a dictionary for {old_label: new_label} mapping, e.g., {1:0, 3:1, 4:2, 5:3, 8:4, ..., 250:158}
            #### such that all labels become 0~158
            base_label_list_raw = base[self.label_key]
            all_base_class = sorted(set(base_label_list_raw))
            print('original labeling:')
            print(all_base_class)
            label_mapping = {}
            for new_lb in range(self.n_class):
                label_mapping[all_base_class[new_lb]] = new_lb
            base_label_list = np.array([label_mapping[old_lb] for old_lb in base_label_list_raw])
            print('new labeling:')
            print(sorted(set(base_label_list)))
            #### split 'base' data into 'train' and 'test' by 9:1
            data_len = len(base_image_list)
            arr_all = np.arange(data_len)
            np.random.shuffle(arr_all)
            self.train_image_list = [base_image_list[i] for i in arr_all[0:int(data_len*0.9)]]
            self.train_label_list = [base_label_list[i] for i in arr_all[0:int(data_len*0.9)]]
            self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
            self.test_image_list = [base_image_list[i] for i in arr_all[int(data_len*0.9):int(data_len)]]
            self.test_label_list = [base_label_list[i] for i in arr_all[int(data_len*0.9):int(data_len)]]
            self.nBatches_test = int(np.ceil(len(self.test_image_list) / self.bsize_test))
            # self.train_image_list = base_image_list
            # self.train_label_list = base_label_list
            # self.nBatches = int(np.ceil(len(self.train_image_list) / self.bsize))
    
    def _parse_function(self, filename, label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.c_dim)
        img = tf.image.resize_images(x_decode, [self.img_size, self.img_size])
        img = tf.cast(img, tf.float32) / 255
        if self.use_aug :
            seed = np.random.randint(0, 2 ** 31 - 1)
            augment_size = self.aug_size #### 224 --> 256; 84 --> 96
            ori_image_shape = tf.shape(img)
            img = tf.image.random_flip_left_right(img, seed=seed)
            img = tf.image.resize_images(img, [augment_size, augment_size])
            img = tf.random_crop(img, ori_image_shape, seed=seed)
            image_mean = tf.constant(self.img_mean, dtype=tf.float32)
            image_std = tf.constant(self.img_std, dtype=tf.float32)
            img = (img - image_mean) / image_std
        return img, label
    
    def build_model(self):
        ### training parameters
        self.bn_train = tf.placeholder('bool')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        ### input data
        self.input_images = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.c_dim], name='input_images')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.labels_vec = tf.one_hot(self.labels, self.n_class)
        
        ### training data flows
        self.feat = self.extractor(self.input_images, self.bn_train, self.with_BN)
        self.logits = self.classifier(self.feat, self.bn_train)
        self.acc = tf.nn.in_top_k(self.logits, self.labels, k=1)
        
        ### collect update operations for moving-means and moving-variances for batch normalizations
        ### [Note] collect before testing operation!
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        ### loss
        self.loss_center, self.centers, self.centers_update_op = self.get_loss_center(self.feat,
                                                                                      self.labels,
                                                                                      self.alpha_center,
                                                                                      self.n_class)
        self.loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_vec,
                                                                              logits=self.logits,
                                                                              name='loss_ce'))
        self.loss = self.loss_ce + self.lambda_center * self.loss_center
        
        with tf.control_dependencies([self.centers_update_op] + self.update_ops):
            if self.used_opt == 'sgd':
                self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif self.used_opt == 'momentum':
                self.opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss)
            elif self.used_opt == 'adam':
                self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=1)

        ### Count number of trainable variables
        total_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape()
            var_params = 1
            for dim in shape:
                var_params *= dim.value
            total_params += var_params
        print('total number of parameters: %d' % total_params)
    
    def get_loss_center(self, features, labels, alpha, num_classes, center_var_name='centers'):
        dim_features = features.get_shape()[1]
        centers = tf.get_variable(center_var_name, [num_classes, dim_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss_center = tf.nn.l2_loss(features - centers_batch)
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        centers_update_op = tf.scatter_sub(centers, labels, diff)
        return loss_center, centers, centers_update_op
    
    def extractor(self, input_images, bn_train, reuse=False):
        raiseNotImplementedError("extractor network not implemented")
    
    def classifier(self, input_feat, bn_train, reuse=False):
        with tf.variable_scope("classifier") as scope:
            if reuse:
                scope.reuse_variables()
            return linear(input_feat, self.n_class, add_bias=True)
    
    def train(self,
              init_from=None,
              lr_start=1e-3,
              lr_decay=0.1,
              lr_decay_step=30):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ### Data indexes used to shuffle training order
        arr = np.arange(len(self.train_image_list))
        
        ### main training loop
        loss_train = []
        acc_train = []
        loss_test = []
        acc_test = []
        best_loss = 0
        stopping_step = 0
        lr_decay_exponent = 0
        start_time = time.time()
        for epoch in range(1, (self.num_epoch+1)):
            lr = lr_start * lr_decay**((epoch-1)//lr_decay_step)
            np.random.shuffle(arr)
            loss_train_batch = []
            acc_train_batch = []
            for idx in range(self.nBatches):
                batch_files = [self.train_image_list[i] for i in arr[idx*self.bsize:(idx+1)*self.bsize]]
                batch = [get_image_resize_normalize(batch_file, self.img_size, random_flip=True, random_crop=True) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_labels = [self.train_label_list[i] for i in arr[idx*self.bsize:(idx+1)*self.bsize]]
                _, loss, acc = self.sess.run([self.opt, self.loss, self.acc],
                                             feed_dict={self.input_images: batch_images,
                                                        self.labels: batch_labels,
                                                        self.bn_train: True,
                                                        self.learning_rate: lr})
                loss_train_batch.append(loss)
                acc_train_batch.append(np.mean(acc))
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))

            #### [2020/07/22] add validation using data specified in 'base_test.json'
            loss_test_batch = []
            acc_test_batch = []
            for idx in range(self.nBatches_test):
                batch_files = self.test_image_list[idx*self.bsize_test:(idx+1)*self.bsize_test]
                batch = [get_image_resize_normalize(batch_file, self.img_size, center_crop=self.center_crop, aug_size=self.aug_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_labels = self.test_label_list[idx*self.bsize_test:(idx+1)*self.bsize_test]
                loss, acc = self.sess.run([self.loss_ce, self.acc],
                                          feed_dict={self.input_images: batch_images,
                                                     self.labels: batch_labels,
                                                     self.bn_train: False})
                loss_test_batch.append(loss)
                acc_test_batch.append(np.mean(acc))
            loss_test.append(np.mean(loss_test_batch))
            acc_test.append(np.mean(acc_test_batch))

            print('Epoch: %d (lr=%f), loss_train: %f, acc_train: %f, loss_test: %f, acc_test: %f' % \
                  (epoch, lr, loss_train[-1], acc_train[-1], loss_test[-1], acc_test[-1]))
        print('time: %4.4f' % (time.time() - start_time))
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        return [loss_train, acc_train, loss_test, acc_test]
    
    def extract(self,
                file_path,
                saved_filename='feature',
                gen_from=None, ## e.g., model_name (must given)
                gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                out_path=None):
        ### Create output folder
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ### Load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")

            #### Load data to be extracted
            with open(file_path, 'r') as reader:
                data = json.loads(reader.read())
            data_image_list = data['image_names']
            nBatches = int(np.ceil(len(data_image_list) / self.bsize))

            features_all = []
            # pred_all = []
            for idx in range(nBatches):
                batch_files = data_image_list[idx*self.bsize:(idx+1)*self.bsize]
                batch = [get_image_resize_normalize(batch_file, self.img_size, center_crop=self.center_crop, aug_size=self.aug_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                feat = self.sess.run(self.feat,
                                     feed_dict={self.input_images: batch_images, self.bn_train: False})
                features_all.append(feat)
                # pred_all.append(pred)
            features_all = np.concatenate(features_all, axis=0)
            # pred_all = np.concatenate(pred_all, axis=0)
            print('features_all.shape: %s' % (features_all.shape,))
            # print('pred_all.shape: %s' % (pred_all.shape,))
            features_dict = {}
            features_dict['features'] = features_all
            features_dict[self.label_key] = data[self.label_key]
            features_dict['image_names'] = data['image_names']
            dopickle(features_dict, os.path.join(self.result_path, self.model_name, saved_filename))
            # return features_all ## return for debug
    
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

class Conv4(BasicCls):
    def __init__(self,
                 sess,
                 model_name='Conv4',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(Conv4, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
            #### x.shape = [-1, 84, 84, 3] or [-1, 32, 32, 3]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h0')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn0')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 42, 42, 64] or [-1, 16, 16, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h1')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn1')
            x = tf.nn.relu(x)           
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h2')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn2')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 10, 10, 64] or [-1, 4, 4, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h3')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn3')
            x = tf.nn.relu()
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 5, 5, 64] or [-1, 2, 2, 64]
            x = tf.contrib.layers.flatten(x)
            #### x.shape = [-1, 1600] or [-1, 256]
            return x

class Conv6(BasicCls):
    def __init__(self,
                 sess,
                 model_name='Conv6',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(Conv6, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
            #### x.shape = [-1, 84, 84, 3] or [-1, 32, 32, 3]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h0')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn0')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 42, 42, 64] or [-1, 16, 16, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h1')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn1')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h2')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn2')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 10, 10, 64] or [-1, 4, 4, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h3')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn3')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            #### x.shape = [-1, 5, 5, 64] or [-1, 2, 2, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h4')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn4')
            x = tf.nn.relu(x)
            #### x.shape = [-1, 5, 5, 64] or [-1, 2, 2, 64]
            x = conv2d(x, output_dim=64, k_h=3, k_w=3, d_h=1, d_w=1, add_bias=(~with_BN), name='h5')
            if with_BN:
                x = batch_norm(x, is_train=bn_train, name='bn5')
            x = tf.nn.relu(x)           
            #### x.shape = [-1, 5, 5, 64] or [-1, 2, 2, 64]
            x = tf.contrib.layers.flatten(x)
            #### x.shape = [-1, 1600] or [-1, 256]
            return x

class ResNet10(BasicCls):
    def __init__(self,
                 sess,
                 model_name='ResNet10',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(ResNet10, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
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
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv3_x
            x = resblk_first(x, out_channel=128, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            #### x.shape = [-1, 11, 11, 128] or [-1, 4, 4, 128]
            #### conv4_x
            x = resblk_first(x, out_channel=256, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            #### x.shape = [-1, 6, 6, 256] or [-1, 2, 2, 256]
            #### conv5_x
            x = resblk_first(x, out_channel=512, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            #### x.shape = [-1, 3, 3, 512] or [-1, 1, 1, 512]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 512]
            return x

class ResNet18(BasicCls):
    def __init__(self,
                 sess,
                 model_name='ResNet18',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(ResNet18, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
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

class ResNet34(BasicCls):
    def __init__(self,
                 sess,
                 model_name='ResNet34',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(ResNet34, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
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
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv2_3')
            #### x.shape = [-1, 21, 21, 64] or [-1, 8, 8, 64]
            #### conv3_x
            x = resblk_first(x, out_channel=128, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_2')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_3')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv3_4')
            #### x.shape = [-1, 11, 11, 128] or [-1, 4, 4, 128]
            #### conv4_x
            x = resblk_first(x, out_channel=256, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_2')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_3')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_4')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_5')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv4_6')
            #### x.shape = [-1, 6, 6, 256] or [-1, 2, 2, 256]
            #### conv5_x
            x = resblk_first(x, out_channel=512, kernels=3, strides=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv5_2')
            x = resblk(x, is_train=bn_train, with_BN=with_BN, name='conv5_3')
            #### x.shape = [-1, 3, 3, 512] or [-1, 1, 1, 512]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 512]
            return x

class ResNet50(BasicCls):
    def __init__(self,
                 sess,
                 model_name='ResNet50',
                 result_path='..',
                 train_path='./json_folder/base_train.json',
                 test_path=None,
                 label_key='image_labels',
                 num_epoch=100,
                 bsize=128,
                 img_size=224,
                 c_dim=3,
                 n_class=64,
                 fc_dim=128,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 alpha_center=0.1,
                 lambda_center=0.0,
                 used_opt='adam',
                 use_aug=True,
                 with_BN=True,
                 center_crop=False):
        super(ResNet50, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    train_path,
                                    test_path,
                                    label_key,
                                    num_epoch,
                                    bsize,
                                    img_size,
                                    c_dim,
                                    n_class,
                                    fc_dim,
                                    bnDecay,
                                    epsilon,
                                    alpha_center,
                                    lambda_center,
                                    used_opt,
                                    use_aug,
                                    with_BN,
                                    center_crop)
    
    def extractor(self, x, bn_train, with_BN=True, reuse=False):
        with tf.variable_scope("ext") as scope:
            if reuse:
                scope.reuse_variables()
            #### x.shape = [-1, 84, 84, 3] or [-1, 224, 224, 3] or [-1, 32, 32, 3]
            #### conv1
            with tf.variable_scope('conv1'):
                x = conv2d(x, output_dim=64, k_h=7, k_w=7, d_h=2, d_w=2, add_bias=(~with_BN))
                #### x.shape = [-1, 42, 42, 64] or [-1, 112, 112, 64] or [-1, 16, 16, 64]
                if with_BN:
                    x = batch_norm(x, is_train=bn_train)
                x = tf.nn.relu(x)
                x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                #### x.shape = [-1, 21, 21, 64] or [-1, 56, 56, 64] or [-1, 8, 8, 64]
            #### conv2_x
            x = convblk_noBN(x, out_channels=[64, 64, 256], stride_first=1, stride_middle=1, is_train=bn_train, with_BN=with_BN, name='conv2_1')
            x = idenblk_noBN(x, out_channels=[64, 64, 256], is_train=bn_train, with_BN=with_BN, name='conv2_2')
            x = idenblk_noBN(x, out_channels=[64, 64, 256], is_train=bn_train, with_BN=with_BN, name='conv2_3')
            #### x.shape = [-1, 21, 21, 256] or [-1, 56, 56, 256] or [-1, 8, 8, 256]
            #### conv3_x
            x = convblk_noBN(x, out_channels=[128, 128, 512], stride_first=1, stride_middle=2, is_train=bn_train, with_BN=with_BN, name='conv3_1')
            x = idenblk_noBN(x, out_channels=[128, 128, 512], is_train=bn_train, with_BN=with_BN, name='conv3_2')
            x = idenblk_noBN(x, out_channels=[128, 128, 512], is_train=bn_train, with_BN=with_BN, name='conv3_3')
            x = idenblk_noBN(x, out_channels=[128, 128, 512], is_train=bn_train, with_BN=with_BN, name='conv3_4')
            #### x.shape = [-1, 11, 11, 512] or [-1, 28, 28, 512] or [-1, 4, 4, 512]
            #### conv4_x
            x = convblk_noBN(x, out_channels=[256, 256, 1024], stride_first=1, stride_middle=2, is_train=bn_train, with_BN=with_BN, name='conv4_1')
            x = idenblk_noBN(x, out_channels=[256, 256, 1024], is_train=bn_train, with_BN=with_BN, name='conv4_2')
            x = idenblk_noBN(x, out_channels=[256, 256, 1024], is_train=bn_train, with_BN=with_BN, name='conv4_3')
            x = idenblk_noBN(x, out_channels=[256, 256, 1024], is_train=bn_train, with_BN=with_BN, name='conv4_4')
            x = idenblk_noBN(x, out_channels=[256, 256, 1024], is_train=bn_train, with_BN=with_BN, name='conv4_5')
            x = idenblk_noBN(x, out_channels=[256, 256, 1024], is_train=bn_train, with_BN=with_BN, name='conv4_6')
            #### x.shape = [-1, 6, 6, 1024] or [-1, 14, 14, 1024] or [-1, 2, 2, 1024]
            #### conv5_x
            x = convblk_noBN(x, out_channels=[512, 512, 2048], stride_first=1, stride_middle=2, is_train=bn_train, with_BN=with_BN, name='conv5_1')
            x = idenblk_noBN(x, out_channels=[512, 512, 2048], is_train=bn_train, with_BN=with_BN, name='conv5_2')
            x = idenblk_noBN(x, out_channels=[512, 512, 2048], is_train=bn_train, with_BN=with_BN, name='conv5_3')
            #### x.shape = [-1, 3, 3, 2048] or [-1, 7, 7, 2048] or [-1, 1, 1, 2048]
            #### global average pooling
            x = tf.reduce_mean(x, [1, 2], name='gap')
            #### x.shape = [-1, 2048]
            return x
