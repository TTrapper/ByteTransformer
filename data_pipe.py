import model_def
import process_config

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import tensorflow as tf


def make_example_generator(filepath, seqlen, batchsize, shuffle):
    total_bytes = os.path.getsize(filepath)
    num_batches = total_bytes // batchsize
    def example_generator():
        starts = range(-batchsize, 0)
        for i in range(num_batches):
            with open(filepath, 'rb') as f:
                maxval = total_bytes - seqlen
                starts = tf.random.uniform([batchsize], minval=0, maxval=maxval,
                        dtype=tf.dtypes.int64) if shuffle else [s + batchsize for s in starts]
                sequences = []
                for start in starts:
                    f.seek(start)
                    sequences.append(f.read(seqlen))
                yield sequences
    return example_generator

def file_to_phrase_dataset(filepath, config, maskinputs=True, shuffle=True):
    batchsize = config['batchsize']
    seqlen = config['seqlen']
    spacing = 0 # max number of bytes between the past and samples (radomly sample)

    example_generator = make_example_generator(filepath, 2 * seqlen + spacing, batchsize, shuffle)
    lines = tf.data.Dataset.from_generator(example_generator, tf.string, tf.TensorShape([batchsize]))
    lines = lines.unbatch()
    lines = lines.map(lambda line: string_to_ids(line))

    lines = lines.map(lambda line: (line[:seqlen], # past/future seperated by random spacing
            tf.slice(line, [tf.random.uniform([], seqlen, 1+seqlen+spacing, tf.int32)], [seqlen])))

    lines = lines.batch(batchsize)
    lines = lines.map(lambda x,y : (tf.reshape(x, [batchsize, seqlen]),
                                    tf.reshape(y, [batchsize, seqlen]))) # explicit shape

    lines = lines.map(lambda x,y : ((x[:, 0:1], x[:, -1:], mask_first_char(mask_last_char(x))),
                                    (y[:, 0:1], y[:, -1:], mask_first_char(mask_last_char(y)))))
    lines = lines.map(lambda x,y : ((tf.concat([x[0], x[1]], axis=1), x[2]),
                                    (tf.concat([y[0], y[1]], axis=1), y[2])))
    if maskinputs:
        lines = lines.map(lambda x,y: ((x[0], randomly_mask_sampled_maskprob(x[1], 0, 0.25)),
                                       (y[0], randomly_mask_sampled_maskprob(y[1], 0, 0.25))))
    lines = lines.prefetch(8)
    return lines


def mask_first_char(tensor):
    batchsize = tensor.shape[0]
    go = tf.zeros([batchsize, 1], tensor.dtype)
    return tf.concat([go, tensor[:, 1:]], axis=1)

def mask_last_char(tensor):
    batchsize = tensor.shape[0]
    stop = tf.zeros([batchsize, 1], tensor.dtype)
    return tf.concat([tensor[:, :-1], stop], axis=1)

def randomly_mask_sampled_maskprob(tensor, min_maskprob, max_maskprob):
    """
    Randomly mask values in the tensor, where the masking rate is uniformly sampled from
    [0, max_maskprob]. This is done independently for each batch item.
    """
    # Sample a masking probabilty for each batch item
    maskprob = tf.random.uniform([tensor.shape[0]], minval=min_maskprob, maxval=max_maskprob,
            dtype=tf.dtypes.float32)
    maskprob = tf.expand_dims(maskprob, axis=1)
    maskprob = tf.tile(maskprob, [1, tensor.shape[1]])
    # Create a mask
    mask = tf.random.uniform(tensor.shape, minval=0, maxval=1, dtype=tf.dtypes.float32)
    mask = tf.where(tf.less(mask, maskprob), tf.zeros_like(mask), tf.ones_like(mask))
    mask = tf.cast(mask, tensor.dtype)
    return tensor * mask#, tensor * (1 - mask)

def string_to_ids(tf_string):
    result = tf.strings.bytes_split(tf_string, 'UTF-8')
    # Decode raw bytes: data is preped to a fixed number of bytes per line so some valid utf-8
    # characters may get split into invalid utf-8 bytes if they lie on the boundary.
    result = tf.io.decode_raw(result, tf.uint8)
    result = tf.cast(result, tf.int32)
    result = tf.squeeze(result, axis=1)
    return result

def ids_to_string(tensor):
    result = tf.strings.unicode_encode(tensor, 'UTF-8', errors='ignore')
    return result

def ids_to_python_string(tensor, maskchar='_'):
    # Manually convert the ints to char bytes, then to a string. This avoids producing weird
    # characters when a unicode sequence has been broken up.
    result = tf.cast(tensor, tf.uint8).numpy()
    result = [str(bytes(line), 'utf-8', 'replace').replace(chr(0), maskchar) for line in result]

    return result

if __name__ == '__main__':
    config = process_config.load_config()
    config['batchsize'] = 10
    lines = file_to_phrase_dataset('./traindata.txt', config)
    print(lines)
    lines = iter(lines)
    (x0, x), (y0, y) = next(lines)
    for i in range(config['batchsize']):
        print(ids_to_python_string(x0[i:i+1]))
        print(ids_to_python_string(x[i:i+1]))
        print(ids_to_python_string(y0[i:i+1]))
        print(ids_to_python_string(y[i:i+1]))
        print()
