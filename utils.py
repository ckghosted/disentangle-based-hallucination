from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import skimage
import skimage.io
import skimage.transform
import scipy.misc
import cv2
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height=64, input_width=64,
            resize_height=64, resize_width=64):
    img = skimage.io.imread(image_path)
    img = img / 255.0
    return img

def get_image_resize(image_path, img_size, center_crop=False, aug_size=256):
    # img = skimage.io.imread(image_path)
    # img = img / 255.0
    # return skimage.transform.resize(img, [img_size, img_size])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if center_crop:
        img = cv2.resize(img, (aug_size, aug_size), interpolation=cv2.INTER_CUBIC)
        # center crop from 256 into 224 ==> (x1, x2) = (y1, y2) = (16, 240)
        x1 = int((aug_size - img_size) / 2)
        x2 = x1 + img_size
        y1 = int((aug_size - img_size) / 2)
        y2 = y1 + img_size
        img = img[x1:x2, y1:y2]
    else:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def get_image_resize_normalize(image_path, img_size, random_flip=False, random_crop=False, center_crop=False, aug_size=256):
    # img = skimage.io.imread(image_path)
    # img = img / 255.0
    # return skimage.transform.resize(img, [img_size, img_size])
    ratio = [3.0/4.0, 4.0/3.0]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if random_flip and np.random.random_sample() > 0.5:
        img = cv2.flip(img, 1)
    if random_crop:
        # https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
        crop_scale = np.random.uniform(0.08, 1.0)
        crop_ratio = np.random.uniform(ratio[0], ratio[1])
        height = img.shape[0]
        width = img.shape[1]
        area = height * width
        for _ in range(10):
            target_area = area * crop_scale
            w = int(round(np.sqrt(target_area * crop_ratio)))
            h = int(round(np.sqrt(target_area / crop_ratio)))
            if 0 < w <= width and 0 < h <= height:
                x1 = np.random.randint(0, height - h + 1)
                x2 = x1 + h
                y1 = np.random.randint(0, width - w + 1)
                y2 = y1 + w
                img = img[x1:x2, y1:y2]
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                return (img - MEAN) / STD
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        x1 = (height - h) // 2
        x2 = x1 + h
        y1 = (width - w) // 2
        y2 = y1 + w
        img = img[x1:x2, y1:y2]
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return (img - MEAN) / STD
    elif center_crop:
        img = cv2.resize(img, (aug_size, aug_size), interpolation=cv2.INTER_CUBIC)
        # center crop from 256 into 224 ==> (x1, x2) = (y1, y2) = (16, 240)
        x1 = int((aug_size - img_size) / 2)
        x2 = x1 + img_size
        y1 = int((aug_size - img_size) / 2)
        y2 = y1 + img_size
        img = img[x1:x2, y1:y2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return (img - MEAN) / STD
    else:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return (img - MEAN) / STD

def get_image_resize_gray(image_path, img_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img = np.expand_dims(img, axis=2)
    return img

# for CUB only
def get_seg_image_resize(image_path, img_size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    seg_path = re.sub('jpg', 'png', re.sub('images', 'segmentations', image_path))
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    seg = cv2.resize(seg, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    seg = seg / 255.0
    seg_tiled = np.tile(np.expand_dims(seg, 2), [1,1,3])
    return img * seg_tiled

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                        'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
            resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]
            B = b.eval()
            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]
            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()
                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}
            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})
                lines += """
                    var layer_%s = {
                    "layer_type": "fc", 
                    "sy": 1, "sx": 1, 
                    "out_sx": 1, "out_sy": 1,
                    "stride": 1, "pad": 0,
                    "out_depth": %s, "in_depth": %s,
                    "biases": %s,
                    "gamma": %s,
                    "beta": %s,
                    "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})
                lines += """
                    var layer_%s = {
                    "layer_type": "deconv", 
                    "sy": 5, "sx": 5,
                    "out_sx": %s, "out_sy": %s,
                    "stride": 2, "pad": 1,
                    "out_depth": %s, "in_depth": %s,
                    "biases": %s,
                    "gamma": %s,
                    "beta": %s,
                    "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                            W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        y_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.y_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_sample})
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            y_sample = np.zeros([config.batch_size, dcgan.y_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_sample})
            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            y = np.random.uniform(-0.2, 0.2, size=(dcgan.y_dim))
            y_sample = np.tile(y, (config.batch_size, 1))
            #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_sample})
            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 3:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            y_sample = np.zeros([config.batch_size, dcgan.y_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            y_sample = np.zeros([config.batch_size, dcgan.y_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]
            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))
        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                        for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
