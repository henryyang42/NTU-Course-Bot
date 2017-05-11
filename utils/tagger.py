import os
import re
import jieba
import numpy as np
from crawler.models import *
from .decorator import run_once

possible_slots = ['title', 'when', 'instructor', 'classroom', 'designated_for', 'required_elective', 'sel_method']


def trim_attr(s):
    s = re.sub(r'\（[^)]*\）', '', s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\-[^)]*', '', s)
    # for _ in range(3):
    #     if s and s[-1] in '一二三四五六上下':
    #         s = s[:-1]
    for rep in ' ()（）：:-「」《》、/+':
        s = s.replace(rep, '')

    return s


@run_once
def jieba_setup():
    # Use zh-tw for better accuracy
    if not os.path.exists('dict.big.txt'):
        os.system("wget %s -O %s" % (
            'https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big',
            'dict.big.txt'))
    jieba.set_dictionary('dict.big.txt')

    print('Adding entities to jieba...')
    entities = []
    for course in Course.objects.filter(semester='105-2'):
        entities.append(trim_attr(course.title))
        entities.append(trim_attr(course.classroom))
        entities.append(course.instructor)
        entities.append(course.designated_for)
        entities.append(course.required_elective)
        # TODO More slot should be added...
    entities = np.unique([entity for entity in entities if entity and ' ' not in entity])

    with open('entity.log', 'w') as f:
        f.write('\n'.join(entities))

    for entity in entities:
        jieba.add_word(entity, freq=99999)

    print('%d entities added.' % len(entities))


def cut(sentence):
    """Return a list of tokens given sentence.
        Entities will be tokenized in this function.
    """
    jieba_setup()
    return list([tok.strip() for tok in jieba.cut(sentence) if tok.strip()])


def BIO(sentence, context):
    inv_context = {v.strip(): k for k, v in context.items()}
    toks = sentence.replace(' ', '')
    toks = cut(toks)
    tags = ['O'] * len(toks)
    for i, tok in enumerate(toks):
        tag = inv_context.get(tok, '')
        if tag in possible_slots:
            tags[i] = 'B_%s' % tag

    return tags
