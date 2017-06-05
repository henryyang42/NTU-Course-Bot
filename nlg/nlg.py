import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from evaluate import evaluate_model, decode_minibatch
from model import Seq2Seq, Seq2SeqAttention
from data_utils import get_minibatch


def predict(semantic_frame):

    #vocab = json.load(open('resource/vocab.json','r'))
    src_word2id = json.load(open('resource/src_word2id.json','r'))
    trg_word2id = json.load(open('resource/trg_word2id.json','r'))
    src_id2word = {v:k for k,v in src_word2id.items()}
    trg_id2word = {v:k for k,v in trg_word2id.items()}

    config = {'training':{}, 'data': {}, 'model':{}}
    config['data']['batch_size'] = 1
    config['data']['max_src_length'] = 20
    config['data']['max_trg_length'] = 40
    config['training']['optimizer'] = 'adam'
    config['training']['lrate'] = 0.0001
    config['model']['dim_word_src'] = 16
    config['model']['dim_word_trg'] = 80
    config['model']['dim'] = 128

    batch_size = config['data']['batch_size']
    src_vocab_size = len(src_word2id)
    trg_vocab_size = len(trg_word2id)

    intent = semantic_frame['diaact']
    if 'request' in semantic_frame['diaact']:
        slots = [intent] + [k for k in list(semantic_frame['request_slots'].keys())]
        slot2word = semantic_frame['request_slots']
    else:
        slots = [intent] + [k for k in list(semantic_frame['inform_slots'].keys())]
        slot2word = semantic_frame['inform_slots']

    src_new = {'data':[slots], 'word2id':src_word2id, 'id2word':src_id2word}
    trg_new = {'data':[['人生','好','困難','到底','該','怎麼辦','呢']], 'word2id':trg_word2id, 'id2word':trg_id2word}
        

    model = Seq2Seq(
            src_emb_dim=config['model']['dim_word_src'],
            trg_emb_dim=config['model']['dim_word_trg'],
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            src_hidden_dim=config['model']['dim'],
            trg_hidden_dim=config['model']['dim'],
            batch_size=batch_size,
            bidirectional=False,
            pad_token_src=src_word2id['<pad>'],
            pad_token_trg=trg_word2id['<pad>'],
            nlayers=1,
            nlayers_trg=1,
            dropout=0.,
            )

    # model is trained on CUDA ... but server doesn't have gpu
    model.load_state_dict(torch.load('models/model.pt', map_location=lambda storage, loc: storage))


    input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(src_new['data'], src_new['word2id'], 0,batch_size, 10, add_start=True, add_end=True)
    input_lines_trg_gold, output_lines_trg_gold, lens_trg, mask_trg = get_minibatch(trg_new['data'], trg_new['word2id'], 0,batch_size, 20, add_start=True, add_end=True)


    # Initialize target with <s> for every sentence
    input_lines_trg = Variable(torch.LongTensor(
        [
            [trg_word2id['<s>']]
            for i in range(input_lines_src.size(0))
        ]
    ))

    input_lines_trg = decode_minibatch(
        config, model, input_lines_src, input_lines_trg, output_lines_trg_gold
    )

    # Copy minibatch outputs to cpu and convert ids to words
    input_lines_trg = input_lines_trg.data.numpy()
    input_lines_trg = [
        [trg_id2word[x] for x in line]
        for line in input_lines_trg
    ]

    output_lines_trg_gold = output_lines_trg_gold.data.numpy()
    output_lines_trg_gold = [
        [trg_id2word[x] for x in line]
        for line in output_lines_trg_gold
    ]

    preds = []
    # Process outputs
    for sentence_pred, sentence_real, sentence_real_src in zip(
        input_lines_trg, output_lines_trg_gold, output_lines_src):
        if '</s>' in sentence_pred:
            index = sentence_pred.index('</s>')
        else:
            index = len(sentence_pred)

        preds.append([slot2word[word] if word in slot2word.keys() else word  for word in sentence_pred[:index + 1]])
        
        """
        print("Predict: {}".format(' '.join(sentence_pred[:index + 1])))
        if '</s>' in sentence_real:
            index = sentence_real.index('</s>')
        else:
            index = len(sentence_real)
        print("RealAns: {}".format(' '.join(['<s>'] + sentence_real[:index + 1])))
        print('===========================================')
        """

    return ''.join(preds[0][1:-1])

if __name__ == '__main__':
    test = {'diaact':'inform', 'inform_slots':{'title':'自然語言處理', 'classroom':'資105', 'when':'星期二'}}
    print(predict(test))
    test = {'diaact':'inform', 'inform_slots':{'title':'智慧對話機器人', 'instructor':'陳縕儂'}}
    print(predict(test))
    test = {'diaact':'inform', 'inform_slots':{'title':'離散數學', 'designated_for':'資訊系','classroom':'資102'}}
    print(predict(test))
    test = {'diaact':'inform', 'inform_slots':{'serial_no':'95046', 'title':'機器學習','instructor':'李宏毅'}}
    print(predict(test))
    test = {'diaact':'request_title', 'request_slots':{}}
    print(predict(test))
    test = {'diaact':'request_instructor', 'request_slots':{}}
    print(predict(test))
    test = {'diaact':'thanks', 'inform_slots':{}}
    print(predict(test))

    print("Bad results :'(")
    test = {'diaact':'inform', 'inform_slots':{'title':'智慧對話機器人', 'required_elective':'選修'}}
    print(predict(test))

