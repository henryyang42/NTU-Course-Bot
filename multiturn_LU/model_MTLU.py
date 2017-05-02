#encoding=utf-8
import os
import time,datetime
import pandas as pd
import numpy as np
import jieba
import keras
from keras.layers import LSTM, GRU,  Embedding, Input, Dense, TimeDistributed, Activation
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Reshape, Masking, Flatten, RepeatVector
from keras.utils import np_utils
from keras import backend as K
#from keras.layers.containers import Graph
#from keras.layers.merge import Dot
import word2vec


np.random.seed(123)  # for reproducibility

########################################################
# to pretrain word2vec
########################################################
def w2v(w2v_dim=100, aspect_review=None, test_review=None) :
    if aspect_review is None :
        print('nooooooo')
        return 0
    polarity_review = pd.read_csv('data/polarity_review_seg.csv')
    test_review = test_review.groupby('Review_id').first()
    reviews = aspect_review['Review'].append(polarity_review['Review']).append(test_review['Review'])
    print('#reviews:', len(reviews))

    with open("data/word2vec_corpus.tmp", "w") as f:
            f.write(("\n".join(reviews)+"\n"))
    print('running word2vec ...')
    word2vec.word2phrase('data/word2vec_corpus.tmp', 'data/word2vec_corpus_phrases', verbose=True)
    word2vec.word2vec('data/word2vec_corpus_phrases', 'data/word2vec_corpus.bin', size=w2v_dim, verbose=True, window=5, cbow=0, binary=1, min_count=1, sample='1e-5', hs=1, iter_=5)

########################################################
# turn NLP input into vector
########################################################
def NLP_to_vec(sentence=None, len_sentence=30, model_w2v=None, dim_w2v=100) :
    lst_seg_sentence = jieba.cut(sentence)
    ary_sentence = np.zeros((len_sentence,dim_w2v))
    t1 = 0
    if lst_seg_sentence != [''] :
        '''
        if len(lst_temp2) > 50 :
            print(i1)
            print(len(lst_temp2))
            print (row)
            #print (lst_temp)
            #print (lst_temp2)
        '''
        for empt in range(len_sentence - len(lst_seg_sentence)) :
            t1 += 1
        for word in lst_seg_sentence :
            if word == '' or word not in model_w2v.vocab : #or word == '' or word == '' or word == '' or word == '海口' or word == '群眾' or word == '中歸' or word == '這也能' or word == '經不住' or word == '' or word == '海天' or  word == '388' or word == '海興' or word == '彭年' or word == '場在' or word == '給凍醒' or word == '借殼'or word == '1702'or word == '1710'  or word == '剛為' or word == '568' or word == '我主':
                if t1 == 0:
                    continue
                else :
                    ary_sentence[t1] = ary_sentence[t1-1]
            else :
                ary_sentence[t1] = model_w2v[word]
            t1 += 1
    ary_sentence = ary_sentence.flatten()
    return ary_sentence

########################################################
# add current sentence into history
########################################################
def history_add(history, sentence_vec, len_history) :
    for i in range(len_history-1) :
        history[i] = history[i+1]
    history[len_history-1] = sentence_vec
    return history

########################################################
# X_train contains history and current sentence
# Y_train is BIO of current sentence
########################################################
def model_MTLU(X_train_history=None, X_train_current=None, len_history=8, len_sentence=30,
               dim_w2v=100, dim_after_rnn=100, num_tag=5, dim_status=8) :
    input_h = Input(shape=(len_history,len_sentence,dim_w2v)) # shape = (?,8,30,100)
    input_c = Input(shape=(1,len_sentence,dim_w2v)) # shape = (?,1,30,100)
    rnn_s2v = GRU(dim_after_rnn)
    m = TimeDistributed(rnn_s2v)(input_h) # shape = (?,8,100)
    u = TimeDistributed(rnn_s2v)(input_c) # shape = (?,1,100)
    p_temp = keras.layers.dot([m,u],axes=-1, normalize=False) # shape = (?,8,1)
    p_r = Reshape((len_history,),input_shape=(len_history,1))(p_temp) # shape = (?,8)
    p_final = K.softmax(p_r) # shape = (?,8)
    p_final_r = Reshape((len_history,1),input_shape=(len_history,))(p_final) # shape = (?,8,1)

    # notice not sure
    h_temp = keras.layers.multiply([p_final_r,m]) # shape = (?,8,100)
    h = K.sum(h_temp, axis=1) # shape = (?,100)
    h_r = Reshape((1,dim_after_rnn),input_shape=(dim_after_rnn,))(h)
    s = keras.layers.add([h_r,u]) # shape = (?,1,100)
    D1 = Dense(256)(s)
    D2 = Dense(128)(D1)
    O = Dense(64)(D2) # shape = (?,1,64)
    c_r = Reshape((len_sentence,dim_w2v),input_shape=(1,len_sentence,dim_w2v))(input_c) # shape = (?,30,100)
    O_r = Reshape((64,),input_shape=(1,64))(O) #shape = (?,30,100)
    O_rp = RepeatVector(len_sentence)(O_r) # shape = (?,30,64)
    final = keras.layers.concatenate([O_rp,c_r]) # shape = (?,30,164)
    Y1 = GRU(num_tag, return_sequences=True, activation='softmax')(final) # shape = (?,?,6)
    Y2 = Dense(dim_status)(final)

    model = Model(inputs=[input_h, input_c], outputs=[Y1,Y2])
    model.compile(#loss='mean_squared_error',
                  loss='categorical_crossentropy',
                  #optimizer='rmsprop',
                  optimizer='adam',
                  metrics=['acc']) #'mae'


    return model

    '''
    p = np.zeros((len(X_train_history),turns_history,1))
    for i in range(len(X_train_history)) :
        p[i] = keras.layers.dot([rnn_h[i],rnn_c], axes=-1)

    temp = keras.layers.multiply([rnn_h,rnn_c])
    p = keras.layers.add(temp,)

    add_input(L_input, input_shape=(8+1,max_len_sentence*w2v_dim), dtype='float')
    add_node(layer, name, input=None, inputs=[], merge_mode='concat', concat_axis=-1, dot_axes=-1, create_output=False)
    '''

########################################################
# return prediction of BIO and new history
########################################################
def run_MTLU(model, history, sentence, len_history) :
    sentence_vec = NLP_to_vec(sentence)
    predictin = model.predict([history,dentence_vec])
    history_new = history_add(history,sentence_vec,len_history)

    return prediction , history_new



if __name__ == '__main__' :
    len_history = 4
    len_sentence = 10
    dim_w2v = 100
    dim_after_rnn = 100
    num_tag = 5 # Nan, O , B_title, B_instructor, B_when
    dim_status = 2*4 # request*4 and constraint*4

    
    ########################################################
    # for test
    ########################################################
    X_train_history = np.random.rand(10,len_history,len_sentence,dim_w2v)
    X_train_current = np.random.rand(10,1,len_sentence,dim_w2v)

    model_mtlu = model_MTLU(X_train_history, X_train_current,
                            len_history=len_history, len_sentence=len_sentence,
                            dim_w2v=dim_w2v, dim_after_rnn=dim_after_rnn,
                            num_tag=num_tag, dim_status=dim_status)

    history = np.zeros((len_history,len_sentence,w2v_dim))
    for i in range(3) :
        temp = np.random.rand(5,dim_w2v)
        history[len_history-1-i][len_sentence-5:len_sentence] = temp

    ########################################################


    ########################################################
    # loading simmulator_log
    # s_log is simmulator_log
    # ['index_sample', 'index_turn', 'sentence', 'BIO', 'status', 'status_for_MTLU']
    s_log = pd.read_csv('./MTLU_template/simmulator_log.csv')
    all_sentence_temp = s_log['sentence']

    temp = s_log['status']
    status = []
    for s in temp :
        s = s.split(' ')
        status_sub = []
        for i in s :
            status_sub.append(int(i))
        status.append(status_sub)
    #print(status)

    temp = s_log['status_for_MTLU']
    status_for_MTLU = []
    for s in temp :
        s = s.split(' ')
        status_sub = []
        for i in s :
            status_sub.append(int(i))
        status_for_MTLU.append(status_sub)
    #print(status_for_MTLU)

    temp = s_log['BIO']
    BIO = []
    for s in temp :
        s = s.split(' ')
        temp2 = []
        for i in s :
            temp2.append(i)
        BIO.append(temp2)
    #print(BIO)
    ########################################################
    X_train_history = np.random.rand(10,len_history,len_sentence,dim_w2v)
    X_train_current = np.random.rand(10,1,len_sentence,dim_w2v)

    model_mtlu = model_MTLU(X_train_history, X_train_current,
                            len_history=len_history, len_sentence=len_sentence,
                            dim_w2v=dim_w2v, dim_after_rnn=dim_after_rnn,
                            num_tag=num_tag, dim_status=dim_status)




    run_MTLU(model_mtlu, history, sentence, len_history)
