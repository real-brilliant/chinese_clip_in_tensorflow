#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  Jeffrey.Sun

import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from utils.bert_huggingface import BertBase
from utils.bert_tokenizer import FullTokenizer
from utils.vit import Transformer, VisualTransformer

current_dir, _ = os.path.split(os.path.realpath(__file__))
img_mean = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
img_var = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


class ModelConfigs(object):
    model_dir = os.path.join(current_dir, 'model')
    model_checkpoint = os.path.join(model_dir, "clip.ckpt")
    vocab_pth = os.path.join(model_dir, "vocab.txt")

    # configs for ViT-32 + mBERT-huggingface
    vocab_size = 119547
    max_seq_len = 512
    txt_hidden_size = 768
    txt_num_heads = 12
    patch_size = 32


tokenizer = FullTokenizer(vocab_file=ModelConfigs.vocab_pth, do_lower_case=False)


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if hasattr(self, "_do_lazy") and not self._do_lazy:
            return func(self)
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property


class TextExample(object):
    """ InputExample for text encoder of UnsupervisedClip """
    def __init__(self, text, max_seq_len=512):
        self.text = text
        self.max_seq_len = max_seq_len

    @lazy_property
    def tokens(self):
        return tokenizer.tokenize(self.text)[0]

    @lazy_property
    def txt_ids(self):
        tokens = ["[CLS]"] + self.tokens[:self.max_seq_len-2] + ["[SEP]"]
        return tokenizer.convert_tokens_to_ids(tokens)


class ImageExample(object):
    """ InputExample for text encoder of UnsupervisedClip """
    def __init__(self, img, from_file=True, input_resolution=224, flip=False):
        self._img = img
        self.from_file = from_file
        self.input_resolution = input_resolution
        self.flip = flip

    @lazy_property
    def img(self):
        img = Image.open(self._img).convert("RGB") if self.from_file else self._img
        if self.flip:
            img = ImageOps.mirror(img)
        return img

    @classmethod
    def img_resize(cls, img):
        width, height = img.size
        ratio = 224 / min(width, height)
        return img.resize((int(width * ratio), int(height * ratio)), resample=Image.BICUBIC)

    @classmethod
    def img_center_crop(cls, img):
        width, height = img.size
        x = int((width - 224) / 2)
        y = int((height - 224) / 2)
        return img.crop((x, y, x+224, y+224))

    @classmethod
    def img_normalize(cls, img):
        return (img - img_mean) / img_var

    @lazy_property
    def img_ids(self):
        img_resize = self.img_resize(self.img)
        img_crop = self.img_center_crop(img_resize)
        img_np = np.array(img_crop) / 255.0
        img_norm = self.img_normalize(img_np)
        return img_norm.transpose(1, 0, 2)


class CLIP(object):
    def __init__(self, vocab_size=49408, max_seq_len=77, patch_size=16, input_resolution=224,
                 txt_num_layers=12, img_num_layers=12, txt_hidden_size=512, img_hidden_size=768, unit_hidden_size=512,
                 txt_num_heads=8, img_num_heads=12):
        self.txt_model = BertBase(
            num_layers=txt_num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=txt_hidden_size,
            num_heads=txt_num_heads,
            unit_hidden_size=unit_hidden_size)
        # self.txt_model = Transformer(
        #     num_layers=txt_num_layers,
        #     vocab_size=vocab_size,
        #     max_seq_len=max_seq_len,
        #     hidden_size=txt_hidden_size,
        #     num_heads=txt_num_heads,
        #     unit_hidden_size=unit_hidden_size)
        self.img_model = VisualTransformer(
            num_layers=img_num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=img_hidden_size,
            num_heads=img_num_heads,
            patch_size=patch_size,
            input_resolution=input_resolution,
            unit_hidden_size=unit_hidden_size)

    def build_model(self, features):
        logit_scale = tf.get_variable(name='logit_scale', shape=[], initializer=tf.ones_initializer, dtype=tf.float32)

        # convert signal into l2_normed features
        self._txt_features = self.txt_model.build(features)  # [Bt, H]
        self._img_features = self.img_model.build(features)  # [Bt, H]
        self.txt_features = tf.nn.l2_normalize(self._txt_features, axis=-1)
        self.img_features = tf.nn.l2_normalize(self._img_features, axis=-1)  # [Bi, H]

        self.img_intermediate = self.img_model.intermediate_embeds  # [B, Layer, L, H]

        # compute cosine_sim
        self.cosine_t2i = tf.matmul(self.txt_features, self.img_features, transpose_b=True) * tf.exp(logit_scale)
        self.cosine_i2t = tf.transpose(self.cosine_t2i, [1, 0])
        self.probs_t2i = tf.nn.softmax(self.cosine_t2i, axis=-1)
        self.probs_i2t = tf.nn.softmax(self.cosine_i2t, axis=-1)

        return self.cosine_t2i, self.cosine_i2t


class Predictor(object):
    """ Predictor for UnsupervisedClip """
    def __init__(self, init_checkpoint):
        self.__init_model(init_checkpoint)

    def __init_model(self, init_checkpoint):
        self.graph = tf.Graph()
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        # gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.sess = tf.Session(config=gpu_config, graph=self.graph)

        with self.graph.as_default():
            self.name2ph = {
                'txt_ids': tf.placeholder(tf.int32, (None, None), 'txt_ids'),
                'img_ids': tf.placeholder(tf.float32, (None, None, None, None), 'img_ids')}
            self.model = CLIP(
                vocab_size=ModelConfigs.vocab_size,
                max_seq_len=ModelConfigs.max_seq_len,
                txt_hidden_size=ModelConfigs.txt_hidden_size,
                txt_num_heads=ModelConfigs.txt_num_heads,
                patch_size=ModelConfigs.patch_size)
            self.model.build_model(features=self.name2ph)

            saver = tf.train.Saver()
            saver.restore(self.sess, init_checkpoint)

    @classmethod
    def _padding(cls, features, pad_val=0):
        max_len = max(map(lambda x: len(x), features))
        return [f + [pad_val] * (max_len - len(f)) for f in features]

    def predict_txt_embeds(self, txt_exms, batch_size=64, l2_norm=True):
        txt_embeds = []
        for i in range(math.ceil(len(txt_exms) / batch_size)):
            feed_dict = {
                self.name2ph['txt_ids']: self._padding([txt_exm.txt_ids for txt_exm in txt_exms[i * batch_size: (i+1) * batch_size]])}
            expected = self.model.txt_features if l2_norm else self.model._txt_features
            txt_embeds.append(self.sess.run(expected, feed_dict))
        return np.concatenate(txt_embeds)

    def predict_img_all_embeds(self, img_exms, batch_size=64, l2_norm=True):
        intermediate_embeds, img_embeds = [], []
        for i in range(math.ceil(len(img_exms) / batch_size)):
            feed_dict = {
                self.name2ph['img_ids']: [img_exm.img_ids for img_exm in img_exms[i * batch_size: (i+1) * batch_size]]}
            _expected = self.model.img_features if l2_norm else self.model._img_features
            _intermediate_embeds, _img_embeds = self.sess.run([self.model.img_intermediate, _expected], feed_dict)
            intermediate_embeds.append(_intermediate_embeds)
            img_embeds.append(_img_embeds)
        return np.concatenate(intermediate_embeds), np.concatenate(img_embeds)

    def predict_img_embeds(self, img_exms, batch_size=64, l2_norm=True):
        img_embeds = []
        for i in range(math.ceil(len(img_exms) / batch_size)):
            feed_dict = {
                self.name2ph['img_ids']: [img_exm.img_ids for img_exm in img_exms[i * batch_size: (i+1) * batch_size]]}
            expected = self.model.img_features if l2_norm else self.model._img_features
            img_embeds.append(self.sess.run(expected, feed_dict))
        return np.concatenate(img_embeds)

    def predict_cosine_sim(self, txt_exms, img_exms):
        feed_dict = {
            self.name2ph['txt_ids']: self._padding([txt_exm.txt_ids for txt_exm in txt_exms]),
            self.name2ph['img_ids']: [img_exm.img_ids for img_exm in img_exms]}
        cosine_t2i, cosine_i2t = self.sess.run([self.model.cosine_t2i, self.model.cosine_i2t], feed_dict)
        return cosine_t2i, cosine_i2t

    def predict_probs(self, txt_exms, img_exms):
        feed_dict = {
            self.name2ph['txt_ids']: self._padding([txt_exm.txt_ids for txt_exm in txt_exms]),
            self.name2ph['img_ids']: [img_exm.img_ids for img_exm in img_exms]}
        probs_t2i, probs_i2t = self.sess.run([self.model.probs_t2i, self.model.probs_i2t], feed_dict)
        return probs_t2i, probs_i2t


if __name__ == "__main__":
    def simple_test():
        import matplotlib.pyplot as plt

        img_files = [
            "./template/green apple.jpg",
            "./template/red apple.jpg",
            "./template/purple apple.png",
            "./template/Orange Apple.png",
            "./template/fruit bowl.jpg",
            "./template/bananas.jpg"]
        img_exms = [ImageExample(img) for img in img_files]

        zh_texts = ['青苹果', '红苹果', '紫苹果', '橙苹果', '一碗水果', '一串香蕉挂在树上']
        txt_exms = [TextExample(txt) for txt in zh_texts]

        def plot_heatmap(result_matrix):
            height, width = result_matrix.shape
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)
            im = ax.imshow(result_matrix)

            # Create X & Y Labels
            ax.set_xticks(np.arange(width))
            ax.set_yticks(np.arange(height))
            ax.set_xticklabels(["Image {}".format(i) for i in range(width)])
            ax.set_yticklabels(["Text {}".format(i) for i in range(height)])

            for i in range(height):
                for j in range(width):
                    text = ax.text(j, i, result_matrix[i, j], ha="center", va="center", color='grey', size=20)

            fig.tight_layout()
            plt.show()

        predictor = Predictor(ModelConfigs.model_checkpoint)
        probs_t2i, probs_i2t = predictor.predict_probs(txt_exms, img_exms)
        plot_heatmap(np.around(probs_t2i, decimals=2).T * 100)

    simple_test()
