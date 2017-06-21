#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json

ap = argparse.ArgumentParser()
ap.add_argument("old_data", type=str, help="old dataset")
ap.add_argument("new_log", type=str, help="new web interface log")
ap.add_argument("out_file", type=str, help="output in the format of training data")
args = ap.parse_args()

sent_set = set()
with codecs.open(args.old_data, "r", "utf-8") as f_old:
    lines = f_old.readlines()
    for i in range(1, len(lines), 3):
        sent = lines[i].strip()
        sent_set.add(sent)

with codecs.open(args.new_log, "r", "utf-8") as f_log, codecs.open(args.out_file, "w", "utf-8") as f_out:
    obj_list = json.load(f_log)
    for obj in obj_list:
        tokens = obj["tokens"]
        sent = u" ".join(tokens)
        if sent in sent_set:
            continue
        sent_set.add(sent)
        intent = obj["intent"]
        labels = obj["labels"]
        f_out.write(intent + "\n")
        f_out.write(sent + "\n")
        f_out.write(u" ".join(labels) + "\n") 
