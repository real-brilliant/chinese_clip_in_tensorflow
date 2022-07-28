#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  Jeffrey.Sun

import math
import numpy as np
import tensorflow as tf

PAD_ID = 0


def variable_scope(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_shape_list(tensor):
    if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
        shape = np.array(tensor).shape
        return shape

    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]

    return shape


def create_initializer(stddev=0.02):
    return tf.truncated_normal_initializer(stddev=stddev)


def gelu(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841
    Args:
        x: float Tensor to perform activation
    Returns:
        `x` with the GELU activation applied.
    """
    x = tf.convert_to_tensor(x)
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))
    return x * cdf


class BertBase(object):
    def __init__(self, num_layers, vocab_size, max_seq_len, hidden_size, num_heads, unit_hidden_size, use_adapters=False):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.att_proj_size = hidden_size // num_heads
        self.unit_hidden_size = unit_hidden_size
        self.use_adapters = use_adapters

    @variable_scope("layer_norm")
    def layer_norm(self, hidden_states, epsilon=1e-12):
        ln_weight = tf.get_variable("weight", [self.hidden_size], initializer=tf.ones_initializer())
        ln_bias = tf.get_variable("bias", [self.hidden_size], initializer=tf.ones_initializer())

        mean = tf.reduce_mean(hidden_states, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.squared_difference(hidden_states, mean), axis=-1, keepdims=True)
        normed_tensor = (hidden_states - mean) * tf.rsqrt(variance + epsilon)

        return normed_tensor * ln_weight + ln_bias

    @variable_scope("adapter")
    def adapter(self, hidden_states, hidden_size, layer_norm=False):
        if layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        _hidden_states = tf.layers.dense(
            hidden_states,
            hidden_size // 2,
            name='in_proj',
            activation=gelu,
            kernel_initializer=create_initializer())
        _hidden_states = tf.layers.dense(
            _hidden_states,
            hidden_size,
            name='out_proj',
            kernel_initializer=create_initializer())
        return _hidden_states

    @variable_scope("attention")
    def multihead_attention(self, hidden_states, attn_mask, batch_size, hidden_size, num_heads, att_proj_size):
        # init weights
        qw = tf.layers.dense(hidden_states, hidden_size, name='q', kernel_initializer=create_initializer())
        kw = tf.layers.dense(hidden_states, hidden_size, name='k', kernel_initializer=create_initializer())
        vw = tf.layers.dense(hidden_states, hidden_size, name='v', kernel_initializer=create_initializer())

        def _reshape(tensor):
            tensor = tf.reshape(tensor, shape=[batch_size, -1, num_heads, att_proj_size])
            return tf.transpose(tensor, [0, 2, 1, 3])

        qw = _reshape(qw)  # [B, num_heads, qL, size_per_head]
        kw = _reshape(kw)  # [B, num_heads, kL, size_per_head]
        vw = _reshape(vw)  # [B, num_heads, kL, size_per_head]

        attn_scores = tf.einsum("bnqd,bnkd->bnqk", qw, kw)  # (B, num_heads, qL, kL)
        attn_scores *= att_proj_size ** -0.5
        if attn_mask is not None:
            attn_scores += attn_mask

        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # (B, n_heads, qL, kL)
        attn_output = tf.matmul(attn_weights, vw)  # [B, n_heads, qL, size_per_head]
        attn_output = tf.reshape(tf.transpose(attn_output, perm=(0, 2, 1, 3)), (batch_size, -1, hidden_size))
        attn_output = tf.layers.dense(attn_output, hidden_size, name='output', kernel_initializer=create_initializer())

        if not self.use_adapters:
            return self.layer_norm(hidden_states + attn_output)
        return self.layer_norm(hidden_states + attn_output + self.adapter(hidden_states, hidden_size))

    @variable_scope("mlp")
    def dense_gelu_dense(self, hidden_states, hidden_size):
        _hidden_states = tf.layers.dense(hidden_states, 4 * hidden_size, name='in_proj', activation=gelu, kernel_initializer=create_initializer())
        _hidden_states = tf.layers.dense(_hidden_states, hidden_size, name='out_proj', kernel_initializer=create_initializer())

        if not self.use_adapters:
            return self.layer_norm(hidden_states + _hidden_states)
        return self.layer_norm(hidden_states + _hidden_states + self.adapter(hidden_states, hidden_size))

    def _encoder(self, hidden_states, attn_mask, batch_size, hidden_size, num_heads, att_proj_size):
        hidden_states = self.multihead_attention(
            hidden_states=hidden_states,
            attn_mask=attn_mask,
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            att_proj_size=att_proj_size)
        hidden_states = self.dense_gelu_dense(
            hidden_states=hidden_states,
            hidden_size=hidden_size)
        return hidden_states

    def body(self, hidden_states, attn_mask, batch_size):
        for idx in range(self.num_layers):
            with tf.variable_scope(f"layer_{idx}"):
                hidden_states = self._encoder(
                    hidden_states=hidden_states,
                    attn_mask=attn_mask,
                    batch_size=batch_size,
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    att_proj_size=self.att_proj_size)
        return hidden_states

    @variable_scope("token_embedding")
    def token_embedding(self, input_ids, seq_len):
        token_embed_table = tf.get_variable(
            name='token_embed_table',
            shape=[self.vocab_size, self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        position_embed_table = tf.get_variable(
            name='position_embed_table',
            shape=[self.max_seq_len, self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        token_type_embed_table = tf.get_variable(
            name='token_type_embed_table',
            shape=[2, self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)

        embeds = tf.gather(token_embed_table, input_ids)
        embeds += tf.expand_dims(position_embed_table[:seq_len, :], axis=0)
        embeds += tf.expand_dims(token_type_embed_table[:1, :], axis=0)

        if not self.use_adapters:
            return self.layer_norm(embeds)
        return self.layer_norm(embeds + self.adapter(embeds, self.hidden_size, True))

    @variable_scope("transformer")
    def build(self, features):
        batch_size, seq_len = get_shape_list(features['txt_ids'])
        _attn_mask = tf.cast(tf.equal(features['txt_ids'], PAD_ID), dtype=tf.float32)
        attn_mask = tf.tile(_attn_mask[:, None, None, :], [1, 1, seq_len, 1]) * -1e9

        txt_embeds = self.token_embedding(features['txt_ids'], seq_len)
        outputs = self.body(txt_embeds, attn_mask, batch_size)

        # slice outputs
        _attn_mask = 1 - _attn_mask
        outputs = tf.reduce_sum(outputs * (_attn_mask[:, :, None]), axis=1) / tf.reduce_sum(_attn_mask, axis=1, keepdims=True)
        # outputs = tf.layers.dense(outputs, self.hidden_size, name='pooler', activation=tf.nn.tanh, kernel_initializer=create_initializer())
        outputs = tf.layers.dense(outputs, self.unit_hidden_size, name='txt_projection', kernel_initializer=create_initializer())
        return outputs
