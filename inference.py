import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import process_config
import numpy as np
import tensorflow as tf
import data_pipe
import random
import model_def



def run_inference(model, input_string, numpredict, temperature=1e-16):
    print('\n******************************************')
    print('softmax temperature: {}'.format(temperature))
    print('******************************************\n')
    temperature = tf.constant(temperature)
    batchsize, seqlen = model.input_shape['past']
    # Convert string to integers and copy for each batch
    input_string = bytes(input_string, 'utf-8')
    input_ids = data_pipe.string_to_ids(tf.constant(input_string))
    input_ids = tf.stack(batchsize * [input_ids])
    result = [input_ids]
    pad = tf.zeros([batchsize, seqlen], input_ids.dtype)
    input_ids = tf.concat([pad, input_ids], axis=1)
    for i in range(numpredict):
        input_ids = tf.concat([input_ids, tf.zeros((batchsize, 1), input_ids.dtype)], axis=1)
        input_ids = input_ids[:, -seqlen:]
        input_ids = data_pipe.mask_first_char(input_ids)

        outputs = model({'past': input_ids, 'future': input_ids})['past']['reconstruct']
        logits = outputs[:, 1, :]
        prediction = tf.random.categorical(logits/temperature, num_samples=1)
        prediction = tf.cast(prediction, input_ids.dtype)
        input_ids = tf.concat([input_ids[:, :-1], prediction], axis=1)
        result.append(prediction)
    # Convert to strings
    outstring = data_pipe.ids_to_python_string(tf.concat(result, axis=1))
    # Print the results for each sequence in the batch
    for line in outstring:
        print(line.replace('\\n', '\n'), '\n')
        print('--------------------------------------------')
    return outstring



if __name__ == '__main__': 
    config = process_config.load_config()

    config['batchsize'] = 4
    config['seqlen'] = 64 #FIXME

    model = model_def.make_phrase_model(config)
    model.load_weights('./phrasemodel.h5', by_name=True, skip_mismatch=True)

    input_string = 'and the meadow was glistening with morning dew as the sun'
    numpredict = 128 
    run_inference(model, input_string, numpredict, 1e-16)
    run_inference(model, input_string, numpredict, 0.5)
    run_inference(model, input_string, numpredict, 0.75)
    run_inference(model, input_string, numpredict, 1.0)
