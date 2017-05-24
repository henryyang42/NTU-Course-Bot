#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json

ap = argparse.ArgumentParser()
ap.add_argument("log_file", type=str, help="web interface log")
ap.add_argument("out_file", type=str, help="output interface")
args = ap.parse_args()

with codecs.open(args.log_file, "r", "utf-8") as f_log, codecs.open(args.out_file, "w", "utf-8") as f_out:
    obj_list = json.load(f_log)
    for obj in obj_list:
        intent = obj["intent"]
        tokens = obj["tokens"]
        labels = obj["labels"]
        f_out.write(intent + "\n")
        f_out.write(u" ".join(tokens) + "\n")
        f_out.write(u" ".join(labels) + "\n") 
