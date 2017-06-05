import json
from random import shuffle

import pandas as pd
import numpy as np

raw_train = json.load(open('data/train.json','r'))
# raw_valid = json.load(open('data/valid.json','r'))
raw_test = json.load(open('data/test.json','r'))
dact = json.load(open('resource/dact.json','r'))
slot_set = dact['intent'] + dact['slots']

config = {'training':{}, 'data': {}, 'model':{}}
config['data']['batch_size'] = 1
config['data']['max_src_length'] = 20
config['data']['max_trg_length'] = 40
config['training']['optimizer'] = 'adam'
config['training']['lrate'] = 0.0001
config['model']['dim_word_src'] = 16
config['model']['dim_word_trg'] = 80
config['model']['dim'] = 128


def read_data(raw_data):

    src_data = []
    trg_data = []

    for i in range(len(raw_data)):
        lhs = raw_data[i][0].find('(')
        rhs = raw_data[i][0].find(')')
        intent = raw_data[i][0][0:lhs]
        semantic = raw_data[i][0][lhs+1:rhs]
        slot_dict = {
                item[1][1:-1]:item[0] 
                for item in [_.split('=') for _ in [pair for pair in semantic.split(';')]] if semantic}

        ### Garbage data ###
        if intent == 'inform' and len(list(slot_dict.keys())) == 0:
            continue

        src_data.append(
                [intent] + 
                [item[0] for item in [_.split('=') for _ in [pair for pair in semantic.split(';')]] if semantic]
        )

        trg_data.append(
                [word if word not in list(slot_dict.keys()) else slot_dict[word] for word in raw_data[i][1].split()])
    
    return src_data, trg_data

###################################################################################################

vocab = [slot for slot in slot_set]
src_data_train, trg_data_train = read_data(raw_train)
src_data_test, trg_data_test = read_data(raw_test)

for line in trg_data_train:
    for word in line:
        if not word in vocab:
            vocab.append(word)
for line in trg_data_test:
    for word in line:
        if not word in vocab:
            vocab.append(word)



src_word2id = {'<s>':0, '<pad>':1, '</s>':2, '<unk>':3}
src_id2word = {0:'<s>', 1:'<pad>', 2:'</s>', 3:'<unk>'}
trg_word2id = {'<s>':0, '<pad>':1, '</s>':2, '<unk>':3}
trg_id2word = {0:'<s>', 1:'<pad>', 2:'</s>', 3:'<unk>'}

for i, slot in enumerate(slot_set):
    src_word2id[slot] = i+4
    src_id2word[i+4] = slot

for i, word in enumerate(vocab):
    trg_word2id[word] = i+4
    trg_id2word[i+4] = word

src_train = {'data':src_data_train, 'word2id':src_word2id, 'id2word':src_id2word}
trg_train = {'data':trg_data_train, 'word2id':trg_word2id, 'id2word':trg_id2word}
src_test = {'data':src_data_test, 'word2id':src_word2id, 'id2word':src_id2word}
trg_test = {'data':trg_data_test, 'word2id':trg_word2id, 'id2word':trg_id2word}


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate import evaluate_model
from model import Seq2Seq, Seq2SeqAttention
from data_utils import get_minibatch

batch_size = config['data']['batch_size']
src_vocab_size = len(src_train['word2id'])
trg_vocab_size = len(trg_train['word2id'])

weight_mask = torch.ones(trg_vocab_size)
weight_mask[trg_train['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)

model = Seq2Seq(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        batch_size=batch_size,
        bidirectional=False,
        pad_token_src=src_train['word2id']['<pad>'],
        pad_token_trg=trg_train['word2id']['<pad>'],
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
    	)
"""
model = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=batch_size,
        bidirectional=False,
        pad_token_src=src_train['word2id']['<pad>'],
        pad_token_trg=trg_train['word2id']['<pad>'],
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
        )
"""

optimizer = optim.Adam(model.parameters(), lr=config['training']['lrate'])


epoch = 1000
for i in range(epoch):

    losses = []

    for j in range(0, len(src_train['data']), batch_size):
        slots = src_train['data'][j][1:]
        shuffle(slots)
        src_train['data'][j][1:] = slots

        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(src_train['data'], src_train['word2id'], j, batch_size, config['data']['max_src_length'])
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(trg_train['data'], trg_train['word2id'], j, batch_size, config['data']['max_trg_length'])


        decoder_logit = model(input_lines_src, input_lines_trg)
        optimizer.zero_grad()


        loss = loss_criterion(
                decoder_logit.contiguous().view(-1, trg_vocab_size),
                output_lines_trg.view(-1)
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()


    print("loss", end='\t')
    print(loss.data[0],end='\t')
    print("remain:", end='\t')
    print(epoch-i)

bleu = evaluate_model(
        model, src_train, src_train, trg_train, trg_train, config,
        verbose=False, metric='bleu'
)
print("Training bleu score:",end='\t')
print(bleu)

bleu = evaluate_model(
        model, src_test, src_test, trg_test, trg_test, config,
        verbose=True, metric='bleu'
)
print("Testing bleu score:",end='\t')
print(bleu)

torch.save(
        model.state_dict(),
        open(os.path.join('models', 'model.pt'),'wb')
)
json.dump(vocab, open('resource/vocab.json','w'), ensure_ascii=False, indent=4)
json.dump(src_word2id, open('resource/src_word2id.json','w'), ensure_ascii=False, indent=4)
json.dump(trg_word2id, open('resource/trg_word2id.json','w'), ensure_ascii=False, indent=4)
