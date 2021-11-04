#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:  GeekFly

import tensorflow as tf

PAD_ID = 0


def variable_scope(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def create_initializer(stddev=0.02):
    return tf.truncated_normal_initializer(stddev=stddev)


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


class Transformer(object):
    def __init__(self, num_layers, vocab_size, max_seq_len, hidden_size, num_heads, unit_hidden_size):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.att_proj_size = hidden_size // num_heads
        self.unit_hidden_size = unit_hidden_size

    @variable_scope("layer_norm")
    def layer_norm(self, hidden_states, epsilon=1e-5):
        ln_weight = tf.get_variable("weight", [self.hidden_size], initializer=tf.ones_initializer())
        ln_bias = tf.get_variable("bias", [self.hidden_size], initializer=tf.ones_initializer())

        mean = tf.reduce_mean(hidden_states, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.squared_difference(hidden_states, mean), axis=-1, keepdims=True)
        normed_tensor = (hidden_states - mean) * tf.rsqrt(variance + epsilon)

        return normed_tensor * ln_weight + ln_bias

    @variable_scope("attention")
    def multihead_attention(self, hidden_states, attn_mask, batch_size, hidden_size, num_heads, att_proj_size):
        # init weights
        in_proj_weight = tf.get_variable(
            name='in_proj_weight',
            shape=[hidden_size, 3 * hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        in_proj_bias = tf.get_variable(
            name='in_proj_bias',
            shape=[3 * hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        out_proj_weight = tf.get_variable(
            name='out_proj_weight',
            shape=[hidden_size, hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        out_proj_bias = tf.get_variable(
            name='out_proj_bias',
            shape=[hidden_size],
            initializer=create_initializer(), dtype=tf.float32)

        _hidden_states = self.layer_norm(hidden_states)
        qkv = tf.matmul(_hidden_states, in_proj_weight, transpose_b=False) + in_proj_bias
        qw, kw, vw = tf.split(qkv, 3, axis=-1)

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

        return hidden_states + tf.matmul(attn_output, out_proj_weight, transpose_b=False) + out_proj_bias

    @variable_scope("mlp")
    def dense_gelu_dense(self, hidden_states, hidden_size):
        in_weight = tf.get_variable(
            name='in_weight',
            shape=[hidden_size, 4 * hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        in_bias = tf.get_variable(
            name='in_bias',
            shape=[4 * hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        out_weight = tf.get_variable(
            name='out_weight',
            shape=[4 * hidden_size, hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        out_bias = tf.get_variable(
            name='out_bias',
            shape=[hidden_size],
            initializer=create_initializer(), dtype=tf.float32)

        _hidden_states = self.layer_norm(hidden_states)
        _hidden_states = tf.matmul(_hidden_states, in_weight, transpose_b=False) + in_bias
        _hidden_states *= tf.sigmoid(1.702 * _hidden_states)
        _hidden_states = tf.matmul(_hidden_states, out_weight, transpose_b=False) + out_bias

        return hidden_states + _hidden_states

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
        return self.layer_norm(hidden_states)

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

        onehot_input_ids = tf.one_hot(input_ids, depth=self.vocab_size, axis=-1)
        output = tf.matmul(onehot_input_ids, token_embed_table)
        return output + tf.expand_dims(position_embed_table[:seq_len, :], axis=0)

    def text_projection(self, txt_features):
        txt_proj_weight = tf.get_variable(
            name='txt_proj_weight',
            shape=[self.hidden_size, self.unit_hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        return tf.matmul(txt_features, txt_proj_weight, transpose_b=False)

    @variable_scope("transformer")
    def build(self, features):
        batch_size, seq_len = get_shape_list(features['txt_ids'])
        seq_ids = tf.range(seq_len)
        attn_mask = tf.less_equal(tf.tile(seq_ids[None, None, :], (batch_size, seq_len, 1)), seq_ids[None, :, None])
        attn_mask = (1.0 - tf.cast(attn_mask, dtype=tf.float32)[:, None, :, :]) * -1e9

        txt_embeds = self.token_embedding(features['txt_ids'], seq_len)
        outputs = self.body(txt_embeds, attn_mask, batch_size)

        # slice outputs
        last_token_idx = tf.reduce_sum(tf.sign(features['txt_ids']), reduction_indices=1) - 1
        gather_idx = tf.stack([tf.range(batch_size), tf.cast(last_token_idx, dtype=tf.int32)], axis=-1)
        outputs = tf.gather_nd(outputs, gather_idx)  # [B, H]

        return self.text_projection(outputs)


class VisualTransformer(Transformer):
    def __init__(self, num_layers, vocab_size, max_seq_len, hidden_size, num_heads,
                 patch_size, input_resolution, unit_hidden_size):
        super(VisualTransformer, self).__init__(num_layers, vocab_size, max_seq_len, hidden_size, num_heads, unit_hidden_size)
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.max_img_len = (input_resolution // patch_size) ** 2 + 1
        self.unit_hidden_size = unit_hidden_size

    @variable_scope("conv")
    def conv_2d(self, input_ids):
        """
        expected shape of 'img_ids' is [B, H, W, C]
            ( and [B, C, W, H] in clip.torch )
        return [B, grid, grid, hidden_size]
            ( and [B, hidden_size, grid, grid] in clip.torch )
        """
        conv_weight = tf.get_variable(
            name='conv_weight',
            shape=[self.patch_size, self.patch_size, 3, self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        return tf.nn.conv2d(input_ids, conv_weight, strides=[1, self.patch_size, self.patch_size, 1], padding='SAME')

    @variable_scope("image_embedding")
    def image_embedding(self, input_ids, batch_size):
        class_embedding = tf.get_variable(
            name='class_embed_table',
            shape=[self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        position_embedding = tf.get_variable(
            name='position_embed_table',
            shape=[self.max_img_len, self.hidden_size],
            initializer=create_initializer(), dtype=tf.float32)

        input_ids = tf.transpose(input_ids, [0, 3, 2, 1])
        input_ids = tf.reshape(input_ids, [batch_size, self.hidden_size, -1])
        input_ids = tf.transpose(input_ids, [0, 2, 1])  # [B, grid ** 2, H]

        embeds = tf.concat([class_embedding + tf.zeros([batch_size, 1, self.hidden_size], dtype=tf.float32), input_ids], axis=1)
        embeds += position_embedding

        return self.layer_norm(embeds)

    def image_projection(self, img_features):
        img_proj_weight = tf.get_variable(
            name='img_proj_weight',
            shape=[self.hidden_size, self.unit_hidden_size],
            initializer=create_initializer(), dtype=tf.float32)
        return tf.matmul(img_features, img_proj_weight, transpose_b=False)

    @variable_scope("visual_transformer")
    def build(self, features):
        batch_size = get_shape_list(features['img_ids'])[0]

        img_ids = self.conv_2d(features['img_ids'])
        img_embeds = self.image_embedding(img_ids, batch_size=batch_size)
        outputs = self.body(hidden_states=img_embeds, attn_mask=None, batch_size=batch_size)

        return self.image_projection(outputs[:, 0, :])
