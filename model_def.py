import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random


class LinearProject(tf.keras.layers.Layer):
    """
    Wraps a linear projection followed by layer normalization and dropout
    """
    def __init__(self, size, dropout, **kwargs):
        super(LinearProject, self).__init__(**kwargs)
        self.dropout = dropout
        self.size = size
        self.dense = tf.keras.layers.Dense(size, None)
        self.layernorm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super(LinearProject, self).get_config()
        config.update({'size':self.size, 'dropout':self.dropout})
        return config

    def call(self, values):
        values = self.layernorm(self.dense(values))
        return tf.keras.layers.Dropout(rate=self.dropout)(values)

class Positioner(tf.keras.layers.Layer):
    """
    Takes a sequence of shape: [batchsize, numitems, itemsize] and adds position information
    """
    def __init__(self, dropout, outsize=None, **kwargs):
        super(Positioner, self).__init__(**kwargs)
        self.dropout = dropout
        self.outsize = outsize

    def get_config(self):
        config = super(Positioner, self).get_config()
        config.update({'dropout':self.dropout})
        return config

    def build(self, input_shape):
        batchsize, nslots, slotsize = input_shape
        self.dense = tf.keras.layers.Dense(self.outsize if self.outsize else slotsize, tf.nn.relu)
        # Fixed position embeds are just the binary representations of the numerical values
        places = 16
        powers = tf.constant([[2 ** i for i in range(places)]])
        pos_embeds = tf.range(nslots, 0, -1) # Count backwards: last item should be most recent
        pos_embeds = tf.tile(tf.expand_dims(pos_embeds, axis=1), [1, places])
        pos_embeds = tf.bitwise.bitwise_and(pos_embeds, powers)
        self.pos_embeds = tf.cast(pos_embeds != 0, tf.float32)
        self.pos_project = tf.keras.layers.Dense(slotsize, None)

    def call(self, values):
        batchsize, nslots, slotsize = values.shape
        pos = self.pos_project(self.pos_embeds)
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batchsize, 1, 1]) # batchsize, nslots, nslots
        values = self.dense(values + pos)
        return tf.keras.layers.Dropout(rate=self.dropout)(values)


class Conv(tf.keras.layers.Layer):
    """
    Transformer with a fixed attention window
    """
    def __init__(self, window, kernelsize, headsize, nheads, dropout, **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.window = window
        self.kernelsize = kernelsize
        self.headsize = headsize
        self.nheads = nheads
        self.dropout = dropout

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update({'window': self.window,
                       'kernelsize': self.kernelsize,
                       'headsize': self.headsize,
                       'nheads': self.nheads,
                       'dropout': self.dropout})
        return config

    def build(self, input_shape):
        self.kqv = tf.keras.layers.Dense(3 * self.nheads * self.headsize)
        self.kernel = tf.keras.layers.Dense(self.kernelsize, tf.nn.relu)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(self.nheads * self.headsize)
        self.droplayer = tf.keras.layers.Dropout(rate=self.dropout)

    def call(self, values):
        batchsize, seqlen, slotsize = values.shape
        nheads = self.nheads
        headsize = self.headsize
        subseqlen = 1 + 2*self.window

        k,q,v = tf.split(self.kqv(values), 3, axis=2)
        q = self.layernorm(q)
        k = self.layernorm(k)
        k,q,v = [tf.reshape(a, [batchsize, seqlen, nheads, headsize]) for a in [k,q,v]]

        indices = []
        for i in range(seqlen):
            start = max(i - self.window, 0)
            start = min(start, seqlen - subseqlen)
            end = start + subseqlen
            indices.append(list(range(start, end)))

        kwindow = tf.gather(k, indices, axis=1)
        vwindow = tf.gather(v, indices, axis=1)
        kwindow = tf.reshape(kwindow, [batchsize * seqlen, subseqlen, nheads, headsize])
        vwindow = tf.reshape(vwindow, [batchsize * seqlen, subseqlen, nheads, headsize])
        qwindow = tf.reshape(q, [batchsize * seqlen, nheads, 1, headsize])
        v_residual = tf.reshape(v, [batchsize * seqlen, 1, nheads * headsize])

        kwindow = tf.transpose(kwindow, [0, 2, 3, 1]) # batch, heads, headsize, seqlen

        attention = tf.matmul(qwindow, kwindow) # batch, heads, 1, k_seqlen
        attention = tf.nn.softmax(attention / (headsize ** 0.5), axis=3)
        attention = tf.transpose(attention, [0, 2, 3, 1]) # bathch, 1, k_seqlen, heads
        attention = tf.reshape(attention, [seqlen * batchsize, 1, subseqlen, nheads, 1])

        vwindow = tf.reshape(vwindow, [seqlen * batchsize, 1, subseqlen, nheads, headsize])
        vwindow *= attention # batchsize, 1, k_seqlen, nheads, headsize
        vwindow = tf.reduce_sum(vwindow, axis=2) # same axis as softmax (after transposes)

        vwindow = tf.reshape(vwindow, [seqlen * batchsize, 1, nheads * headsize])
        vwindow = self.droplayer(self.kernel(vwindow))
        vwindow = self.droplayer(self.project(vwindow))
        vwindow += v_residual


        return tf.reshape(vwindow, [batchsize, seqlen, slotsize])




class Transformer(tf.keras.layers.Layer):
    def __init__(self, kernelsize, headsize, nheads, dropout, dilate=False, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.kernelsize = kernelsize
        self.headsize = headsize
        self.nheads = nheads
        self.dropout = dropout
        self.dilate = dilate
        self.activations = {}

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({'kernelsize':self.kernelsize,
                       'dropout':self.dropout})
        return config

    def build(self, input_shape):
        self.kqv = tf.keras.layers.Dense(3 * self.nheads * self.headsize)
        self.kernel = tf.keras.layers.Dense(self.kernelsize, tf.nn.relu)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.project = tf.keras.layers.Dense(self.nheads * self.headsize)

    def call(self, values):
        batchsize, seqlen, slotsize = values.shape
        nheads = self.nheads
        headsize = self.headsize

        k,q,v = tf.split(self.kqv(values), 3, axis=2)
        q = self.layernorm(q)
        k = self.layernorm(k)
        v_residual = v
        k,q,v = [tf.reshape(a, [batchsize, seqlen, nheads, headsize]) for a in [k,q,v]]

        if self.dilate: # use every nth query
            dialation = 8 # FIXME replace self.dilate with a configurable dilation amound
            assert nheads % dialation == 0
            heads = tf.split(v_residual, dialation, axis=2)
            for i in range(dialation):
                heads[i] = tf.gather(heads[i], range(i, seqlen, dialation), axis=1)
            v_residual = tf.concat(heads, axis=2)

            heads = tf.split(q, dialation, axis=2)
            for i in range(dialation):
                heads[i] = tf.gather(heads[i], range(i, seqlen, dialation), axis=1)
            q = tf.concat(heads, axis=2)

        k = tf.transpose(k, [0, 2, 3, 1]) # batch, heads, headsize, seqlen
        q = tf.transpose(q, [0, 2, 1, 3]) # batch, heads, seqlen, headsize

        num_queries = q.shape[2]
        attention = tf.matmul(q, k) # batch, heads, q_seqlen, k_seqlen
        attention = tf.nn.softmax(attention / (headsize ** 0.5), axis=3)
        self.activations['attention'] = attention
        attention = tf.transpose(attention, [0, 2, 3, 1]) # bathch, q_seqlen, k_seqlen, heads
        attention = tf.reshape(attention, [batchsize, num_queries, seqlen, nheads, 1])

        v = tf.reshape(v, [batchsize, 1, seqlen, nheads, headsize])
        v *= attention # batchsize, q_seqlen, k_seqlen, nheads, headsize
        v = tf.reduce_sum(v, axis=2) # same axis as softmax (after transposes)

        v = tf.reshape(v, [batchsize, num_queries, nheads * headsize])
        v = self.kernel(v)
        v = tf.keras.layers.Dropout(self.dropout)(v)
        v = self.project(v)
        v = tf.keras.layers.Dropout(self.dropout)(v)
        return v + v_residual


class Trainables(tf.keras.layers.Layer):
    """
    Exposes a tensor of trainable variable
    """
    def __init__(self, n_embeds, **kwargs):
        super(Trainables, self).__init__(**kwargs)
        self.n_embeds = n_embeds

    def get_config(self):
        config = super(Trainables, self).get_config()
        config.update({'n_embeds':self.n_embeds})
        return config

    def build(self, input_shape):
        batchsize, seqlen, slotsize = input_shape
        self.trainables = self.add_weight(shape=(1, self.n_embeds, slotsize), name='weights')

    def call(self, values):
        batchsize, seqlen, slotsize = values.shape
        tiled = tf.tile(self.trainables, [batchsize, 1, 1])
        return tf.concat([tiled, values], axis=1)

def make_phrase_model(config, output_attention=False):
    inputs_collector = {}
    outputs_collector = []
    dropout = config['dropout']
    headsize = config['headsize']
    batchsize = config['batchsize']
    seqlen = config['seqlen']
    char_embed_size = config['char_embed_size']
    train_char_embeds = config['train_char_embeds']
    activations = []
    # Define layers used by the model
    char_embed_layer = tf.keras.layers.Embedding(config['numclasses'], config['char_embed_size'],
            trainable=train_char_embeds)
    char_layer_norm = tf.keras.layers.LayerNormalization(trainable=train_char_embeds)
    discriminator = []
    discriminator.append(tf.keras.layers.Dense(1024, tf.nn.relu))
    discriminator.append(tf.keras.layers.Dense(1024, tf.nn.relu))
    discriminator.append(tf.keras.layers.Dense(1024, tf.nn.relu))
    discriminator.append(tf.keras.layers.Dense(1024, tf.nn.relu))
    decision_layer = tf.keras.layers.Dense(1, None)

    def make_inputs(name):
        char_ids = tf.keras.Input(shape=(seqlen), batch_size=batchsize, name=name)
        inputs_collector[name] = char_ids
        return char_ids

    def embeddings(name):
        char_ids = make_inputs(name)
        char_embeds = char_embed_layer(char_ids)# + char_embed_layer(char_ids_normalized)
        return char_layer_norm(char_embeds)

    encoder_layers = []
    encoder_layers.append(Conv(window=2, kernelsize=1024, headsize=64, nheads=4, dropout=dropout))
    encoder_layers.append(Conv(window=2, kernelsize=1024, headsize=64, nheads=4, dropout=dropout))
    encoder_layers.append(Conv(window=2, kernelsize=1024, headsize=64, nheads=4, dropout=dropout))
    encoder_layers.append(Conv(window=2, kernelsize=1024, headsize=64, nheads=4, dropout=dropout))
    encoder_layers.append(tf.keras.layers.Reshape([seqlen // 8, 4 * 64 * 8]))
    encoder_layers.append(tf.keras.layers.Dense(2048, tf.nn.relu))
    encoder_layers.append(Trainables(3)) # Concatenate special tokens to use for predictions
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))
    encoder_layers.append(Transformer(2048, headsize=64, nheads=16, dropout=dropout, dilate=False))

    position_layer = Positioner(dropout=0.0)
    reconstruct_layer = tf.keras.layers.Dense(256, name='reconstruct_logits')
    # Build inputs and run them through the layers, which are reused for each input stream
    outputs = {}
    for stream_name in ['past', 'future']:
        char_embeds = embeddings(stream_name)
        char_embeds = position_layer(char_embeds)
        layer_outs = []
        layer_reconstructs = []
        for layer in encoder_layers:
            char_embeds = layer(char_embeds)
            batchsize, new_seqlen, slotsize = char_embeds.shape
        activations.append(layer_outs)
        activations.append(layer_reconstructs)
        # FIXME assuming location of special tokens
        class_tokens = char_embeds[:, :1] # Predict past/future ordering
        recon_tokens = char_embeds[:, 1:3] # Predict next/previous bytes
        outputs[stream_name] = {'vector':class_tokens,
                                'reconstruct':reconstruct_layer(recon_tokens)}

    ##########################################

    # Combine streams and decide which pairs are true (past, future) and which are false
    true_pairs = tf.concat([outputs['past']['vector'], outputs['future']['vector']], axis=1)
    false_pairs = tf.concat([outputs['future']['vector'], outputs['past']['vector']], axis=1)
    # Final layers responsible for deciding between the true and false pairs
    droplayer = tf.keras.layers.Dropout(rate=dropout)
    for layer in discriminator:
        true_pairs = droplayer(layer(true_pairs))
        false_pairs = droplayer(layer(false_pairs))
    true_logits = decision_layer(true_pairs)
    false_logits = decision_layer(false_pairs)

    print(true_logits, false_logits)
    print(inputs_collector)
    outputs['true_logits'] = true_logits
    outputs['false_logits'] = false_logits
    outputs['activations'] = activations
    print(outputs)
    model = tf.keras.Model(inputs=inputs_collector, outputs=outputs)
    return model
