#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json

ap = argparse.ArgumentParser()
ap.add_argument("log_file", type=str, help="web interface log")
ap.add_argument("out_file", type=str, help="output in the format of training data")
ap.add_argument("-u", "--unique", action="store_true", help="unique utterance")
args = ap.parse_args()

with codecs.open(args.log_file, "r", "utf-8") as f_log, codecs.open(args.out_file, "w", "utf-8") as f_out:
    sent_set = set()
    obj_list = json.load(f_log)
    for obj in obj_list:
        tokens = obj["tokens"]
        sent = u" ".join(tokens)
        if args.unique and sent in sent_set:
            continue
        sent_set.add(sent)
        intent = obj["intent"]
        labels = obj["labels"]
        f_out.write(intent + "\n")
        f_out.write(sent + "\n")
        f_out.write(u" ".join(labels) + "\n") 
