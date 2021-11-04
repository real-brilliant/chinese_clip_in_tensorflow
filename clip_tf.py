#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  GeekFly

import tensorflow as tf

from vit.visual_transformer import Transformer, VisualTransformer


class TextExample(object):
    def __init__(self, txt_ids):
        self.txt_ids = txt_ids


class ImageExample(object):
    def __init__(self, img_ids):
        self.img_ids = img_ids


class CLIP(object):
    def __init__(self, vocab_size=49408, max_seq_len=77, patch_size=16, input_resolution=224,
                 txt_num_layers=12, img_num_layers=12, txt_hidden_size=512, img_hidden_size=768, unit_hidden_size=512,
                 txt_num_heads=8, img_num_heads=12):
        self.txt_model = Transformer(
            num_layers=txt_num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=txt_hidden_size,
            num_heads=txt_num_heads,
            unit_hidden_size=unit_hidden_size)
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

        # convert signal to l2_normed features
        self.txt_features = tf.nn.l2_normalize(self.txt_model.build(features), axis=-1)  # [Bt, H]
        self.img_features = tf.nn.l2_normalize(self.img_model.build(features), axis=-1)  # [Bi, H]

        # compute cosine_sim
        self.cosine_t2i = tf.matmul(self.txt_features, self.img_features, transpose_b=True) * tf.exp(logit_scale)
        self.cosine_i2t = tf.transpose(self.cosine_t2i, [1, 0])

        return self.cosine_t2i, self.cosine_i2t


class Predictor(object):
    def __init__(self, init_checkpoint):
        self.__init_model(init_checkpoint)

    def __init_model(self, init_checkpoint):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            self.name2ph = {
                'txt_ids': tf.placeholder(tf.int32, (None, None), 'txt_ids'),
                'img_ids': tf.placeholder(tf.float32, (None, None, None, None), 'img_ids')}
            self.model = CLIP()
            self.model.build_model(features=self.name2ph)

            saver = tf.train.Saver()
            saver.restore(self.sess, init_checkpoint)

    def _padding(self, features, pad_val=0):
        max_len = max(map(lambda x: len(x), features))
        return [f + [pad_val] * (max_len - len(f)) for f in features]

    def predict_txt_embeds(self, txt_exms):
        pass

    def predict_img_embeds(self, img_exms):
        pass

    def predict_cosine_sim(self, txt_exms, img_exms):
        feed_dict = {
            self.name2ph['txt_ids']: self._padding([txt_exm.txt_ids for txt_exm in txt_exms]),
            self.name2ph['img_ids']: [img_exm.img_ids for img_exm in img_exms]}

        cosine_t2i, cosine_i2t = self.sess.run([self.model.cosine_t2i, self.model.cosine_i2t], feed_dict)

        return cosine_t2i, cosine_i2t
