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

class seq2seq():
    def __init__(self):

        self.config = {'training':{}, 'data': {}, 'model':{}}
        self.config['data']['batch_size'] = 1
        self.config['data']['max_src_length'] = 20
        self.config['data']['max_trg_length'] = 40
        self.config['training']['optimizer'] = 'adam'
        self.config['training']['lrate'] = 0.0001
        self.config['model']['dim_word_src'] = 16
        self.config['model']['dim_word_trg'] = 80
        self.config['model']['dim'] = 128

        self.src_word2id = json.load(open('resource/src_word2id.json','r'))
        self.src_id2word = {v:k for k,v in self.src_word2id.items()}
        self.trg_word2id = json.load(open('resource/trg_word2id.json','r'))
        self.trg_id2word = {v:k for k,v in self.trg_word2id.items()}

        self.src_vocab_size = len(self.src_word2id)
        self.trg_vocab_size = len(self.trg_word2id)

        self.model = Seq2Seq(
                src_emb_dim=self.config['model']['dim_word_src'],
                trg_emb_dim=self.config['model']['dim_word_trg'],
                src_vocab_size=self.src_vocab_size,
                trg_vocab_size=self.trg_vocab_size,
                src_hidden_dim=self.config['model']['dim'],
                trg_hidden_dim=self.config['model']['dim'],
                batch_size=1,
                bidirectional=False,
                pad_token_src=self.src_word2id['<pad>'],
                pad_token_trg=self.trg_word2id['<pad>'],
                nlayers=1,
                nlayers_trg=1,
                dropout=0.,
        )

        self.model.load_state_dict(torch.load('models/model.pt', map_location=lambda storage, loc: storage))

    def predict(self, semantic_frame):


        intent = semantic_frame['diaact']
        if 'request' in semantic_frame['diaact']:
            slots = [intent] + [k for k in list(semantic_frame['request_slots'].keys())]
            slot2word = semantic_frame['request_slots']
        else:
            slots = [intent] + [k for k in list(semantic_frame['inform_slots'].keys())]
            slot2word = semantic_frame['inform_slots']

        src_new = {'data':[slots], 'word2id':self.src_word2id, 'id2word':self.src_id2word}
        trg_new = {'data':[['人生','好','困難','到底','該','怎麼辦','呢']], 'word2id':self.trg_word2id, 'id2word':self.trg_id2word}

        preds = []

        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(src_new['data'], src_new['word2id'], 0, 1, 10, add_start=True, add_end=True)
        input_lines_trg_gold, output_lines_trg_gold, lens_trg, mask_trg = get_minibatch(trg_new['data'], trg_new['word2id'], 0, 1, 20, add_start=True, add_end=True)


        # Initialize target with <s> for every sentence
        input_lines_trg = Variable(torch.LongTensor(
            [
                [self.trg_word2id['<s>']]
                for i in range(input_lines_src.size(0))
            ]
        ))

        input_lines_trg = decode_minibatch(
            self.config, self.model, input_lines_src, input_lines_trg, output_lines_trg_gold
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_trg = input_lines_trg.data.cpu().numpy()
        input_lines_trg = [
            [self.trg_id2word[x] for x in line]
            for line in input_lines_trg
        ]

        output_lines_trg_gold = output_lines_trg_gold.data.cpu().numpy()
        output_lines_trg_gold = [
            [self.trg_id2word[x] for x in line]
            for line in output_lines_trg_gold
        ]

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

    nlg = seq2seq()

    test = {'diaact':'inform', 'inform_slots':{'title':'自然語言處理', 'classroom':'資105', 'when':'星期二'}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'inform', 'inform_slots':{'title':'智慧對話機器人', 'instructor':'陳縕儂'}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'inform', 'inform_slots':{'title':'智慧對話機器人', 'required_elective':'選修'}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'inform', 'inform_slots':{'title':'離散數學', 'designated_for':'資訊系','classroom':'資102'}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'inform', 'inform_slots':{'serial_no':'95046', 'title':'機器學習','instructor':'李宏毅'}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'request_title', 'request_slots':{}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'request_instructor', 'request_slots':{}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'thanks', 'inform_slots':{}}
    print(test)
    print(nlg.predict(test))
    test = {'diaact':'inform', 'inform_slots':{'classroom':'資101'}}
    print(test)
    print(nlg.predict(test))

