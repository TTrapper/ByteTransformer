import json

def load_config():
    path = './config.json'
    config = json.load(open(path))
    config['numclasses'] = 256 # assume utf-8 bytes
    print_config(config)
    return config

def print_config(config):
    print('CONFIGURATION')
    print('batchsize: {}'.format(config['batchsize']))
    print('context window: {}'.format(config['seqlen']))
    print('headsize: {}'.format(config['headsize']))
    print('char_embed_size: {}'.format(config['char_embed_size']))
    print('dropout: {}'.format(config['dropout']))
    print('train_char_embeds: {}'.format(config['train_char_embeds']))


if '__main__' == __name__:
    load_config()
