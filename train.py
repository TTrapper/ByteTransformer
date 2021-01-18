import argparse
import os
import time

import numpy as np
import tensorflow as tf

import data_pipe
import model_def
import process_config

import bokeh
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, TapTool, CustomJS, LabelSet

import umap

parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--traindata', default='./traindata.txt', help='Path to a text file')
parser.add_argument('--validdata', default='./validdata.txt', help='Path to a text file')
parser.add_argument('--evaldata', default='./evaldata.txt', help='Path to a text file')


def parseargs(parser):
    args = parser.parse_args()
    if args.eval:
        raise NotImplementedError('Eval mode not yet set up')
    if not args.eval and not os.path.isfile(args.traindata):
        raise FileNotFoundError('No training data file found at {}'.format(args.traindata))
    if not args.eval and not os.path.isfile(args.validdata):
        print('No validation data file found at: {}. Continuing without it.'.format(args.validdata))
        args.validdata = None
    if args.eval and not os.path.isfile(args.evaldata):
        raise FileNotFoundError('No eval data file found at {}'.format(args.traindata))
    return args

def train(restore, datapath):
    learn_rate = 1e-4
    config = process_config.load_config()
    model = model_def.make_phrase_model(config)
    model.summary()
    if restore:
        model.load_weights('./phrasemodel.h5', by_name=True, skip_mismatch=True)
    dataset = data_pipe.file_to_phrase_dataset(datapath, config, maskinputs=True, shuffle=True)
    random_dataset = data_pipe.file_to_phrase_dataset(datapath, config, maskinputs=True, shuffle=True)
    dataset = iter(dataset)
    random_dataset = iter(dataset)
    optimizer = tf.keras.optimizers.Adam(learn_rate)
    binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    reconstruct_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    order_mean = tf.keras.metrics.Mean()
    recon_mean = tf.keras.metrics.Mean()
    moving_average = []
    for b, (example, random_example) in enumerate(zip(dataset, random_dataset)):
        past, future = example
        inputs = {'past': past[1],
                  'future': future[1]}
        # Optimize the model
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            true_logits = outputs['true_logits']
            false_logits = outputs['false_logits']
            past_recon = outputs['past']['reconstruct']
            future_recon = outputs['future']['reconstruct']
            # Loss for predicting which pair is ordered correctly (past/future vs future/past)
            loss_value = binary_loss(tf.ones_like(true_logits), true_logits)
            loss_value += binary_loss(tf.zeros_like(false_logits), false_logits)
            order_mean.update_state(loss_value)
            # Loss for predicting the next and previous bytes for both the past and future sequences
            recon_loss_value = reconstruct_loss(past[0], past_recon)
            recon_loss_value += reconstruct_loss(future[0], future_recon)
            recon_loss_value *= 0.5
            recon_mean.update_state(recon_loss_value)
            loss_value += recon_loss_value
            # Moving average of the total loss
            moving_average.append(loss_value)
            moving_average = moving_average[-1000:]
            # Compute and apply gradients
            grads = tape.gradient(loss_value, model.trainable_variables)
#        for g, v in zip(grads, model.trainable_variables):
#            print(tf.reduce_min(g).numpy(), tf.reduce_max(g).numpy(), v.name)
#        print('_________________')
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Track progress
        if b % 1000 == 200:
            model.save('./phrasemodel.h5', save_format='h5', overwrite=True, include_optimizer=False)
        # Print progress
        print('{} order:{:.4}   byte-modeling:{:.4}   total(moving av):{:.4}\r'.format(
            b, order_mean.result().numpy(), recon_mean.result().numpy(),
            sum(moving_average)/len(moving_average)), end='', flush=True)

def phrase_embeds(datapath):
    config = process_config.load_config()
    config['batchsize'] = 1024
    model = model_def.make_phrase_model(config)
    model.load_weights('./phrasemodel.h5', by_name=True, skip_mismatch=True)
    dataset = data_pipe.file_to_phrase_dataset(datapath, config, maskinputs=False, shuffle=True)
    dataset = iter(dataset)
    numbatches = 50
    results = []
    labels = []
    for b in range(numbatches):
        print(b)
        past,future = next(dataset)
        inputs = {'past': past, # FIXME dataset returns tuples when masking enabled
                  'future': future}
        labels.extend(data_pipe.ids_to_python_string(past)) #FIXME past should be a tuple
        out = model.predict_on_batch(inputs)
        results.append(out['past']['vector'])
    labels = [l.replace(chr(0), '_') for l in labels]
    labels = ['{} : {}'.format(l, l[-1]) for l in labels]
    np.save('./embed_labels', np.array(labels))
    results = np.concatenate(results, axis=0)
    np.save('./embeds', results)
    print('saved embeds')
    projected_embeds = umap.UMAP().fit_transform(results)
    np.save('embeds_projected', projected_embeds)
    plot_projections(projected_embeds, labels)
    exit()


def plot_projections(projection, labels):
    # output to static HTML file
    output_file('embeds.projected.html')
    source = ColumnDataSource(data=dict(
        x=projection[:, 0],
        y=projection[:, 1],
        desc=labels))
    # Custom JS that runs when a point is clicked
    code = """
    const data = labels.data
    for (i=0; i<points.selected.indices.length; i++) {
        const ind = points.selected.indices[i]
        currentIdx = data.t.indexOf(points.data.desc[ind])
        if (currentIdx > -1) {
            data.x.splice(currentIdx, 1)
            data.y.splice(currentIdx, 1)
            data.t.splice(currentIdx, 1)
            currentIdx = data.ind.indexOf(ind)
            data.ind.splice(currentIdx, 1)
            continue;
        } else {
            data.x.push(points.data.x[ind])
            data.y.push(points.data.y[ind])
            data.t.push(points.data.desc[ind])
            data.ind.push(ind)
        }
    }
    console.log(data)
    labels.change.emit()
    """
    p = figure(title='embeds', plot_width=1000, plot_height=1000)
    p.title.text_font_size = '16pt'
    # Hover tool
    hover = HoverTool(tooltips=[("", "@desc")])
    p.add_tools(hover)
    labels = ColumnDataSource(data=dict(x=[], y=[], t=[], ind=[]))
    # Tap tool (run custom JS)
    callback=CustomJS(args=dict(points=source, labels=labels), code=code)
    tap = TapTool(callback=callback)
    p.add_layout(LabelSet(x='x', y='y', text='t', y_offset=4, x_offset=4, source=labels))
    p.add_tools(tap)
    p.circle('x', 'y', source=source, size=8)
    show(p)


if '__main__' == __name__:
    args = parseargs(parser)
    datapath = args.evaldata if args.eval else args.traindata

    train(args.restore, datapath)
    phrase_embeds(datapath)
