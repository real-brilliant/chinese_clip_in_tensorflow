#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  GeekFly

import re
import os
import numpy as np
import tensorflow as tf

import clip
import torch


def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
    tf_var = tf.get_variable(
        dtype=tf.dtypes.as_dtype(tensor.dtype), shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
    session.run(tf.variables_initializer([tf_var]))
    session.run(tf_var)
    return tf_var


def var_name_mapping(torch_var_name):
    if 'positional_embedding' in torch_var_name:
        return "transformer/token_embedding/position_embed_table"
    if 'token_embedding' in torch_var_name:
        return "transformer/token_embedding/token_embed_table"
    if 'ln_final' in torch_var_name:
        return re.sub("ln_final.", "transformer/layer_norm/", torch_var_name)

    name = re.sub("resblocks.", "layer_", torch_var_name)
    name = re.sub("\.", "/", name)

    if 'attn' in name:
        name = re.sub("attn", 'attention', name)
        name = re.sub("out_proj/", "out_proj_", name)
    if 'mlp' in name:
        name = re.sub("c_fc/weight", 'in_weight', name)
        name = re.sub("c_fc/bias", 'in_bias', name)
        name = re.sub("c_proj/weight", 'out_weight', name)
        name = re.sub("c_proj/bias", 'out_bias', name)
    elif 'ln_1' in name:
        name = re.sub("ln_1", "attention/layer_norm", name)
    elif 'ln_2' in name:
        name = re.sub("ln_2", "mlp/layer_norm", name)

    return name


def visual_name_mapping(torch_var_name):
    name = re.sub("resblocks.", "layer_", torch_var_name)
    name = re.sub("\.", "/", name)

    if 'class_embedding' in torch_var_name:
        return 'visual_transformer/image_embedding/class_embed_table'
    elif 'positional_embedding' in torch_var_name:
        return 'visual_transformer/image_embedding/position_embed_table'
    elif 'conv' in torch_var_name:
        return 'visual_transformer/conv/conv_weight'
    elif 'ln_pre' in torch_var_name:
        return re.sub("visual/ln_pre", "visual_transformer/image_embedding/layer_norm", name)
    elif 'ln_post' in torch_var_name:
        return re.sub("visual/ln_post", "visual_transformer/layer_norm", name)

    if 'attn' in name:
        name = re.sub("attn", 'attention', name)
        name = re.sub("out_proj/", "out_proj_", name)
    if 'mlp' in name:
        name = re.sub("c_fc/weight", 'in_weight', name)
        name = re.sub("c_fc/bias", 'in_bias', name)
        name = re.sub("c_proj/weight", 'out_weight', name)
        name = re.sub("c_proj/bias", 'out_bias', name)
    elif 'ln_1' in name:
        name = re.sub("ln_1", "attention/layer_norm", name)
    elif 'ln_2' in name:
        name = re.sub("ln_2", "mlp/layer_norm", name)

    return re.sub("visual/transformer", "visual_transformer", name)


def transpose_var_val(tf_var_name):
    if 'in_proj_weight' in tf_var_name:
        return True
    if 'out_proj_weight' in tf_var_name:
        return True
    if 'in_weight' in tf_var_name:
        return True
    if 'out_weight' in tf_var_name:
        return True
    return False


def _convert(pytorch_model, tf_model_pth, tf_model_name):
    if not os.path.isdir(tf_model_pth):
        os.makedirs(tf_model_pth)

    state_dict = pytorch_model.state_dict()

    tf.reset_default_graph()
    with tf.Session() as session:
        for torch_var_name in state_dict:
            torch_var_val = state_dict[torch_var_name].numpy()
            if torch_var_name == 'text_projection':
                tf_var_name = 'transformer/txt_proj_weight'
            elif torch_var_name == 'visual.proj':
                tf_var_name = 'visual_transformer/img_proj_weight'
            elif 'visual' in torch_var_name:
                # convert_name
                tf_var_name = visual_name_mapping(torch_var_name)
                # do transpose
                if transpose_var_val(tf_var_name):
                    torch_var_val = torch_var_val.T
                elif 'conv' in tf_var_name:
                    torch_var_val = torch_var_val.transpose(3, 2, 1, 0)
            else:
                # convert_name
                tf_var_name = var_name_mapping(torch_var_name)
                # do transpose
                if transpose_var_val(tf_var_name):
                    torch_var_val = torch_var_val.T

            # create tf_var
            tf_var = create_tf_var(tensor=torch_var_val, name=tf_var_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_var_val)
            tf_var_val = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_var_name, np.allclose(tf_var_val, torch_var_val)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(tf_model_pth, tf_model_name.replace("-", "_").replace(".ckpt", "") + ".ckpt"))


def convert(pytorch_model_pth="model/ViT-B-16.pt", tf_model_pth="", tf_model_name=""):
    model, _ = clip.load(pytorch_model_pth, device='cpu')
    _convert(pytorch_model=model, tf_model_pth=tf_model_pth, tf_model_name=tf_model_name)


if __name__ == '__main__':
    convert(
        tf_model_pth="clip_in_tf",
        tf_model_name="clip_en")
