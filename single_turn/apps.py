from django.apps import AppConfig
from django.conf import settings
from LU_LSTM.lstm_predict import *
import jieba

class SingleTurnConfig(AppConfig):
    name = 'single_turn'

    # Run only once on start.
    def ready(self):

        if not settings.DEBUG:  # Only load model in production to speed up debugging.
            jieba.load_userdict('%s/crawler/entity_dictionary.txt' % settings.BASE_DIR)
            global lu_model, idx2label, idx2intent, word2idx

            # load vocab
            obj = json.load(
                open('%s/LU_LSTM/re_seg.1K+log_extend_1000.vocab.json' % settings.BASE_DIR, "r"))
            idx2label = obj["slot_vocab"]
            idx2intent = obj["intent_vocab"]
            word2idx = {}
            for i, w in enumerate(obj["word_vocab"]):
                word2idx[w] = i

            # load model
            lu_model = load_model(
                '%s/LU_LSTM/PY3--re_seg.1K+log_extend_1000--LSTM.model' % settings.BASE_DIR)
            print('[Info] Single-turn LU model loaded.')

        else:
            print('[Info] Under DEBUG mode, single-turn LU is not loaded.')
