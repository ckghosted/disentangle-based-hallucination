import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

import model_ext_placeholder as model
model_dict = dict(Conv4=model.Conv4,
                  Conv6=model.Conv6,
                  ResNet10=model.ResNet10,
                  ResNet18=model.ResNet18,
                  ResNet34=model.ResNet34,
                  ResNet50=model.ResNet50)

def main():
    parser = argparse.ArgumentParser()
    ## model parameters
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--model_name', type=str, help='Folder name to save the model (under result_path), must start with Conv4/Conv6/ResNet10/ResNet18/ResNet34/ResNet50')
    parser.add_argument('--n_class', default=921, type=int, help='Number of classes')
    parser.add_argument('--c_dim', default=3, type=int, help='Number of channels of the input image (1 or 3)')
    parser.add_argument('--fc_dim', default=1024, type=int, help='Dimension of the hidden layer in the classifier (not used if only a single layer with softmax)')
    parser.add_argument('--alpha_center', default=0.1, type=float, help='Center update rate')
    parser.add_argument('--lambda_center', default=0.0, type=float, help='Center loss weight')
    parser.add_argument('--used_opt', default='adam', type=str, help='Used optimizer (sgd/momentum/adam)')
    ## training parameters
    parser.add_argument('--train_path', type=str, help='Path of the training json file')
    parser.add_argument('--test_path', default='None', type=str, help='Path of the testing json file')
    parser.add_argument('--label_key', default='image_labels', type=str, help='Label key name in the json files, normally image_labels or image_labels_id')
    parser.add_argument('--num_epoch', default=100, type=int, help='Max number of training epochs')
    parser.add_argument('--bsize', default=128, type=int, help='Batch size')
    parser.add_argument('--lr_start', default=1e-3, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', default=10, type=int, help='Number of epochs per learning rate decay')
    parser.add_argument('--use_aug', action='store_true', help='Add basic image augmentation if present')
    parser.add_argument('--with_BN', action='store_true', help='Use batch normalization if present')
    parser.add_argument('--run_extraction', action='store_true', help='Extract features and saved as pickle files if present')
    parser.add_argument('--center_crop', action='store_true', help='Center crop images for validation and feature extraction if present')
    
    args = parser.parse_args()
    ## Extract the used model from the specified model_name (start with Conv4/Conv6/ResNet10/ResNet18/ResNet34/ResNet50)
    m = re.search('(.+?)_', args.model_name)
    if m:
        used_model = m.group(1)
        if used_model in model_dict.keys():
            print('Use %s' % used_model)
            img_size = int(re.search('img(.+?)_', args.model_name).group(1)) #### Image size, normally 32, 64, 128, or 224
            train(args, used_model, img_size)
            #### Extract features and saved as pickled files for few-shot multiclass classification experiments
            if args.run_extraction:
                extract(args, used_model, img_size)
        else:
            print('Model %s not defined' % used_model)
    else:
        print('Cannot extract used_model from model_name: %s' % args.model_name)
    
def train(args, used_model, img_size):
    print('============================ train ============================')
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        net = model_dict[used_model](sess,
                                     model_name=args.model_name,
                                     result_path=args.result_path,
                                     train_path=args.train_path,
                                     test_path=args.test_path,
                                     label_key=args.label_key,
                                     num_epoch=args.num_epoch,
                                     bsize=args.bsize,
                                     img_size=img_size,
                                     c_dim=args.c_dim,
                                     n_class=args.n_class,
                                     fc_dim=args.fc_dim,
                                     lambda_center=args.lambda_center,
                                     used_opt=args.used_opt,
                                     use_aug=args.use_aug,
                                     with_BN=args.with_BN,
                                     center_crop=args.center_crop)
        net.build_model()
        results = net.train(lr_start=args.lr_start,
                            lr_decay=args.lr_decay,
                            lr_decay_step=args.lr_decay_step)

    np.save(os.path.join(args.result_path, args.model_name, 'results.npy'), results)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
    ax[0].plot(range(1, len(results[2])+1), results[2], label='Validation error')
    ax[0].set_xticks(np.arange(start=0, stop=len(results[0]), step=10))
    ax[0].set_xlabel('Training epochs', fontsize=16)
    # ax[0].set_ylabel('Cross entropy', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[0].grid(True)
    ax[1].plot(range(1, len(results[1])+1), results[1], label='Training accuracy')
    ax[1].plot(range(1, len(results[3])+1), results[3], label='Validation accuracy')
    ax[1].set_xticks(np.arange(start=0, stop=len(results[1]), step=10))
    ax[1].set_xlabel('Training epochs', fontsize=16)
    # ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].legend(fontsize=16)
    ax[1].grid(True)
    plt.suptitle('Learning Curve', fontsize=20)
    fig.savefig(os.path.join(args.result_path, args.model_name, 'learning_curve.png'),
                bbox_inches='tight')
    plt.close(fig)

def extract(args, used_model, img_size):
    print('============================ extract ============================')    
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    
    training_file = os.path.basename(args.train_path)
    training_file_dir = os.path.dirname(args.train_path)
    if training_file == 'base_train.json':
        file_to_extract = ['base_train.json', 'base_test.json', 'val_train.json', 'val_test.json', 'novel_train.json', 'novel_test.json']
    elif training_file == 'base.json':
        file_to_extract = ['base.json', 'val.json', 'novel.json']
    else:
        assert 0
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        net = model_dict[used_model](sess,
                                     model_name=args.model_name,
                                     result_path=args.result_path,
                                     train_path=args.train_path,
                                     test_path=args.test_path,
                                     label_key=args.label_key,
                                     num_epoch=args.num_epoch,
                                     bsize=args.bsize*5,
                                     img_size=img_size,
                                     c_dim=args.c_dim,
                                     n_class=args.n_class,
                                     fc_dim=args.fc_dim,
                                     lambda_center=args.lambda_center,
                                     used_opt=args.used_opt,
                                     use_aug=args.use_aug,
                                     with_BN=args.with_BN)
        net.build_model()
        for f in file_to_extract:
            print('---------------------------- [%s] ----------------------------' % f)
            net.extract(file_path=os.path.join(training_file_dir, f),
                        saved_filename=re.sub('.json', '_feat', f),
                        gen_from=os.path.join(args.result_path, args.model_name, 'models'),
                        out_path=os.path.join(args.result_path, args.model_name))

if __name__ == '__main__':
    main()
